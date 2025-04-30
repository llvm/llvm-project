@Library('nextci@master') _
import groovy.transform.Field
import org.nextsilicon.Builder
import org.nextsilicon.Nextrunner

/* This drives next-llvm-project multi-branch pipeline. */

// Releases configuration
@Field String DEBIAN12_RELEASE = "debian-12"
@Field String ROCKY_RELEASE = "rocky-9"

// This one considered "main release branch"
// We would like to have it rocky-9, however:
// On release branches, only one ("primary") runs
// See https://jenkins.k8s.nextsilicon.com/job/next-llvm-project/
// This needs to run "Code Style" which needs to install clang-format-17
// This is only possible on debian-12 now (not on rocky 9.2)
// Rocky 9.4 might make it possible to run clang-format-17
// When we upgrade to Rocky 9.4 we may return PRIMARY_RELEASE to ROCKY_RELEASE

@Field String PRIMARY_RELEASE = DEBIAN12_RELEASE

// cing configuration
@Field String JOB_TYPE = "pr-head"
@Field String LIMIT_OUTPUT = "1000"
@Field String TOOLCHAIN_TARGET_FOLDER = "linux_next_home"
@Field String TOOLCHAIN_TARGET_PATH = "/workspace/${TOOLCHAIN_TARGET_FOLDER}"
@Field String CING_VERSION = "main"

properties([[$class: 'JiraProjectProperty'],
    buildDiscarder(
        logRotator(artifactDaysToKeepStr: '',
            artifactNumToKeepStr: '',
            daysToKeepStr: '30',
            numToKeepStr: '20')),
    parameters([
        string(name: 'WARNINGS_FILTER_FILE',
            defaultValue: 'warnings_filter_rules.json',
            description: 'Warnings filter rules filename')
    ]),
])


def stashToolchain(String osVariant) {
    if (currentBuild.currentResult != 'SUCCESS' || !env.CHANGE_ID) { return }

    stage("Archive toolchain") {
        sh(script: """cing-copy-artifacts -limit-output ${LIMIT_OUTPUT} -job-type ${JOB_TYPE} \
            -dst ${TOOLCHAIN_TARGET_PATH} \
            -src ${env.NEXT_HOME} \
            -artifact-file ${env.WORKSPACE}/packages/artifacts.txt""",
            label: "Copy next-llvm-project artifacts execution")

        sh(script: """cing-artifactory -limit-output ${LIMIT_OUTPUT} -job-type ${JOB_TYPE} upload \
            -src-artifact ${TOOLCHAIN_TARGET_PATH} \
            -dst-artifact generic-repo/raw-llvm/${osVariant}/${TOOLCHAIN_TARGET_FOLDER}.tar.xz-${env.BRANCH_NAME} \
            -dst-artifact generic-repo/raw-llvm/${osVariant}/${TOOLCHAIN_TARGET_FOLDER}.tar.xz-${nextDocker.getToolchainVersion()}-${env.BRANCH_NAME}""",
            label: "Upload next-llvm-project tar to artifactory")
    }
}

def build(Builder builder, String osVariant) {
    sshagent(['github-ssh-key']) {
        sh(script: "ssh-keyscan -H github.com >> ~/.ssh/known_hosts",
            label: 'set ssh known hosts')

        stage('Checkout') {
            retry(3) {
                // Pass the branch twice - both as branch for checkout and as target branch for rebase. This makes rebase degenrated.
                builder.checkoutRepo(env.BRANCH_NAME, env.BRANCH_NAME)
            }
        }

        stage('Build Setup') {
            // Exec Cing builder-setup
            sh(script: "cing-builder-setup -limit-output ${LIMIT_OUTPUT} -job-type ${JOB_TYPE} next-llvm-project", label: "Cing exec builder-setup")
        }

        if (osVariant == PRIMARY_RELEASE) {
            println "Doing code style"
            warnError("Code Style") {
                stage("Code Style") {
                    // Note: this only works on debian
                    sh(script: """echo "deb http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-17 main" >> /etc/apt/sources.list
                        apt-get -q update
                        apt-get install -y clang-format-17""",
                        label: "Add git-clang-format")
                    String refBase = ""
                    if (env.CHANGE_ID) {
                        echo "Base/Target branch: ${pullRequest.base} <- Head/PR branch: ${pullRequest.headRef}"
                        refBase = "origin/${pullRequest.base}"
                    }
                    def clangFormatStatus = sh(script: "nextsilicon/clang-format.sh ${refBase}",
                                              label: "Style check with clang format",
                                              returnStatus: true)
                    if (clangFormatStatus != 0) {
                        unstable("clang format found errors!")
                    }
                }
            }
        }
        stage('Build') {
            sh(script: "cing-builder -job-type ${JOB_TYPE} next-llvm-project", label: "Cing exec builder")
        }
    }
}

/* run toolchain tests */
def runTests(Builder builder, String osVariant) {
  sshagent(['github-ssh-key']) {
    if (env.CHANGE_ID) {
        // For PRs ONLY
        if (osVariant != PRIMARY_RELEASE) {
            warnError("LLVM lit tests failed") {
                stage("LLVM lit tests") {
                    dir(WORKSPACE) {
                        sh(script: """. /opt/venv/bin/activate
                          ./run_llvm_lit_tests.sh ${WORKSPACE}/${Builder.RELEASE}/llvm""",
                            label: "Run llvm lit tests")
                    }
                }
            }
        } else {
            warnError("LLVM ninja check failed") {
                stage("LLVM ninja check") {
                    dir("${WORKSPACE}/${Builder.RELEASE}/llvm") {
                        sh(script: """. /opt/venv/bin/activate
                            ninja check-llvm""",
                            label: "Run llvm ninja check")
                    }
                }
            }

            warnError("LLVM ninja check-clang failed") {
                stage("LLVM ninja check-clang") {
                    dir("${WORKSPACE}/${Builder.RELEASE}/llvm") {
                        sh(script: """. /opt/venv/bin/activate
                            ninja check-clang""",
                            label: "Run llvm ninja check-clang")
                    }
                }
            }
        }
    }

    // For PRs and branches - run OpenMP
    if (osVariant != PRIMARY_RELEASE) {
        warnError("OpenMP next tests failed") {
            stage("OpenMP next tests") {
                dir("${WORKSPACE}/nextsilicon") {
                    sh(script:""". /opt/venv/bin/activate
                        ./run_openmp_nextsilicon_tests.sh --release""",
                        label: "Run OpenMP next tests")
                }
            }
        }
    }
  }
}

/* report compiler warnings */
def compilerWarnings(Builder builder) {
    warnError("Compiler warnings") {
      stage("Compiler warnings") {
        builder.reportBuildWarnings(threshold: 1,
            referenceBuild: 'next-llvm-project/next_release_170',
            filterFile: params.WARNINGS_FILTER_FILE)
        }
    }
}

def prepareStage(Builder builder, String osVariant) {
    String nodeCpu = "16"
    String nodeMemory = "32"
    String nodeCpuLimit = "40"
    String nodeMemoryLimit = "32"

    def stage = {
        println "==================== SETUP FROM HERE ON ===================="
        def osImage = "${env.JFROG_CONTAINER_REGISTRY}/nextbuilder-${osVariant}:${env.NEXTBUILDER_IMAGE_TAG}"
        nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-${osVariant}", osImage,
                nodeCpu, nodeMemory, nodeCpuLimit, nodeMemoryLimit, CING_VERSION) {
            println "==================== SETUP FINISHED ===================="
            build(builder, osVariant)
            runTests(builder, osVariant)
            stashToolchain(osVariant)
            if (!env.CHANGE_ID || osVariant == PRIMARY_RELEASE) {
                compilerWarnings(builder)
            }
        }
    }
    return stage
}

/*Support PR head single run, Kill all previous job of current PR*/
nextK8s.abortPreviousBuilds()

timestamps {
    ansiColor('xterm') {
        def Builder builder = new Builder(ctx: this, basebranch: 'next_release_170')

        println "==================== SETUP FROM HERE ON ===================="
        /* run PR head with lightweight container to free resources when build container is done */
        def osImage = "${env.JFROG_CONTAINER_REGISTRY}/nextbuilder-${PRIMARY_RELEASE}:${env.NEXTBUILDER_IMAGE_TAG}"
        nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-main", osImage, "500m", "1", "1", "1", CING_VERSION) {
            println "==================== SETUP FINISHED ===================="
            withEnv(["NEXT_HOME=/opt/nextsilicon"]) {
                // for PRs, build Rocky and Deb12.
                // For branches, only build Rocky.
                String[] os_variants = [ROCKY_RELEASE,]
                // OLD:  RHEL_RELEASE, DEBIAN_RELEASE]
                if (env.CHANGE_ID) {
                    os_variants += DEBIAN12_RELEASE
                }
                Map builders = [failFast: true]
                os_variants.each { osVariant ->
                    String osVariantName = osVariant
                    builders[osVariantName] = prepareStage(builder, osVariantName)
                }
                parallel builders
            }
        }
    }//ansiColor
}//timestamps
