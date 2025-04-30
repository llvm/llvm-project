@Library('nextci@master') _

import groovy.transform.Field
import org.nextsilicon.Builder
import org.nextsilicon.NextSlack

/* This drives toolchain-pre-merge pipeline. */

// cing configuration
@Field String JOB_TYPE = "pre-merge"
@Field String LIMIT_OUTPUT = "1000"
@Field String CING_VERSION = "main"
@Field String TOOLCHAIN_TARGET_FOLDER = "linux_next_home"
@Field String TOOLCHAIN_TARGET_PATH = "/workspace/${TOOLCHAIN_TARGET_FOLDER}"

@Field String DEBIAN12_RELEASE = "debian-12"
@Field String ROCKY_RELEASE = "rocky-9"
@Field String PRIMARY_RELEASE = ROCKY_RELEASE

@NonCPS
Boolean userBuildCause() {
  def buildCauses = currentBuild.rawBuild.getCauses()
  for (buildcause in buildCauses) {
    if (buildcause.class == hudson.model.Cause$UserIdCause) {
      return true
    }
  }
  return false
}

/* build toolchain */
void buildToolchain(Builder toolchain, String osVariant) {
  int buildTimeout = 120
  timeout(time: buildTimeout, unit: 'MINUTES') {
    stage("Build ${toolchain.projectName}") {
      echo "Building ${toolchain.projectName} on ${osVariant} with ${buildTimeout} minutes timeout"
      sh(script: "mkdir -p ${TOOLCHAIN_TARGET_PATH} || true", label: "create build dir")
      sshagent(['github-ssh-key']) {
        dir("${env.WORKSPACE}/${toolchain.projectName}") {
          // Checkout next-llvm-project */
          toolchain.checkoutRepo(params.TOOLCHAIN_BRANCH, // source branch
                                "", // do not rebase
                                true, // include change log
                                true, // include poll
                                false, // include submodules
                                true, // submodules shallow clone
                                true, // no tags
                                false) // do not force branch checkout
          /* Exec cing builder-setup */
          sh(script: "cing-builder-setup -limit-output ${LIMIT_OUTPUT} -job-type ${JOB_TYPE} next-llvm-project", label: "Cing exec builder-setup")
          /* Exec cing builder */
          sh(script: """cing-builder -job-type ${JOB_TYPE} ${toolchain.projectName}""",
            label: "Build and install release toolchain")
        }
      }
    }
  }
}

/* stash toolchain artifacts - can be used for debug */
void stashToolchain(Builder toolchain, String osVariant) {

  stage("Stash ${toolchain.projectName}") {
    echo "Stashing ${toolchain.projectName} artifacts for ${osVariant}"
    sh(script: """cing-copy-artifacts -limit-output ${LIMIT_OUTPUT} \
      -dst ${TOOLCHAIN_TARGET_PATH} \
      -src ${env.NEXT_HOME} \
      -artifact-file ${env.WORKSPACE}/${toolchain.projectName}/packages/artifacts.txt""",
      label: "Copy ${toolchain.projectName} artifacts execution")
    sh(script: """
      cing-artifactory -limit-output ${LIMIT_OUTPUT} -job-type ${JOB_TYPE} upload \
      -src-artifact ${TOOLCHAIN_TARGET_PATH} \
      -dst-artifact generic-repo/raw-llvm/${osVariant}/${TOOLCHAIN_TARGET_FOLDER}.tar.xz-${nextDocker.getToolchainVersion()}-${JOB_BASE_NAME}""",
      label: "Upload ${toolchain.projectName} tar to artifactory")
  }
}

/* tag release artifacts */
void tagArtifacts(String osVariant, Boolean autoMerge) {

  if (currentBuild.currentResult != 'SUCCESS') {
    return
  }

  stage("Tag artifacts") {
    String toolchainTag = nextDocker.getToolchainVersion()
    if (!autoMerge) {
      toolchainTag += "-${JOB_BASE_NAME}-test"
    }
    sh(script: """
      cing-artifactory -limit-output ${LIMIT_OUTPUT} -job-type ${JOB_TYPE} copy \
      -src-artifact generic-repo/raw-llvm/${osVariant}/${TOOLCHAIN_TARGET_FOLDER}.tar.xz-${nextDocker.getToolchainVersion()}-${JOB_BASE_NAME} \
      -dst-artifact generic-repo/raw-llvm/${osVariant}/${TOOLCHAIN_TARGET_FOLDER}.tar.xz-${toolchainTag}""",
      label: "Tag release tar file for ${osVariant}")
  }
}

/* run toolchain tests */
void testToolchain(Builder toolchain, String osVariant) {
    sshagent(['github-ssh-key']) {
        if (osVariant == PRIMARY_RELEASE) {
          warnError("LLVM lit tests failed") {
            stage("LLVM lit tests") {
              dir("${env.WORKSPACE}/${toolchain.projectName}") {
                sh(script: """. /opt/venv/bin/activate
                  ./run_llvm_lit_tests.sh ${WORKSPACE}/${toolchain.projectName}/${Builder.RELEASE}/llvm""",
                  label: "Run llvm lit tests")
              }
            }
          }
        } else {
          warnError("LLVM ninja check failed") {
            stage("LLVM ninja check") {
              dir("${env.WORKSPACE}/${toolchain.projectName}/${Builder.RELEASE}/llvm") {
                sh(script: """. /opt/venv/bin/activate
                  ninja check-llvm""",
                  label: "Run llvm ninja check")
              }
            }
          }

          warnError("LLVM ninja check-clang failed") {
            stage("LLVM ninja check-clang") {
              dir("${env.WORKSPACE}/${toolchain.projectName}/${Builder.RELEASE}/llvm") {
                sh(script: """. /opt/venv/bin/activate
                  ninja check-clang""",
                  label: "Run llvm ninja check-clang")
              }
            }
          }
        }

      if (osVariant == PRIMARY_RELEASE) {
        dir("${env.WORKSPACE}/${toolchain.projectName}/nextsilicon") {
          warnError("OpenMP next tests failed") {
            stage("OpenMP next tests") {
              sh(script:""". /opt/venv/bin/activate
                ./run_openmp_nextsilicon_tests.sh --release""",
                label: "Run OpenMP next tests")
            }
          }
        }
      }
    }
}

/*
 * Auto bump toolchain version in case of next-llvm-project PR merge:
 * Bump toolchain version commit is pushed into next-llvm-project versioning branch
 * called bump_toolchain_version at the beginning of the pre merge pipeline.
 * In case of a successful pre merge pipeline, we push the version commit into next-llvm-project PR.
 * Toolchain versioning levels - MAJOR.MINOR.PATCHLEVEL (X.Y.Z):
 * 1. MAJOR (X) - major version update (usually requires code changes in nextutils repository).
 * 2. MINOR (Y) - minor version update (usually requires code changes in nextutils repository).
 * 3. PATCHLEVEL (Z) - patch level version update.
 */
void bumpToolchain(Builder toolchain, String runType, Boolean autoMerge, Boolean merge) {
    String bumpCmd = """/opt/venv/bin/python3 bump_toolchain_version.py \
        --update-type ${env.BUMP_TOOLCHAIN_LEVEL} \
        --toolchain-path ${env.WORKSPACE}/${toolchain.projectName}"""

    if (!merge) {
        bumpCmd += " --bump --branch bump_toolchain_version_${runType}_branch --push --force"
    } else {
        bumpCmd += " --tag --toolchain-sha ${toolchain.getGitSha().trim()}"
        if (autoMerge) {
            bumpCmd += " --push --force"
        }
    }

    /* Lightweight checkout next-llvm-project */
    dir("${env.WORKSPACE}/${toolchain.projectName}") {
        toolchain.checkoutRepo(env.TOOLCHAIN_BRANCH, // source branch
            env.TOOLCHAIN_TARGET, // rebase branch
            true, // include change log
            true, // include poll
            true, // disable submodules
            true, // submodules shallow clone
            true, // no tags
            merge) // force branch checkout
    }

    /* Bump toolchain version */
    sshagent(['github-ssh-key']) {
        dir("${env.WORKSPACE}/${toolchain.projectName}/scripts") {
            sh(script: "${bumpCmd}", label: "Bump toolchain version")
        }
    }

    if (!merge) {
        dir("${env.WORKSPACE}/${toolchain.projectName}") {
            toolchain.updateGitSha()
        }
    }
}

def prepareStage(Builder toolchain, String osVariant, Boolean autoMerge) {
    String nodeCpu = "16"
    String nodeMemory = "16"
    String nodeCpuLimit = "32"
    String nodeMemoryLimit = "32"

    def stage = {
        def osImage = "${env.JFROG_CONTAINER_REGISTRY}/nextbuilder-${osVariant}:${env.NEXTBUILDER_IMAGE_TAG}"
        echo "${osImage}"
        println "==================== SETUP FROM HERE ON ===================="
        nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-llvm-build-${osVariant}",
                osImage, nodeCpu, nodeMemory, nodeCpuLimit, nodeMemoryLimit, CING_VERSION) {
            println "==================== SETUP FINISHED ===================="
            buildToolchain(toolchain, osVariant)
            stashToolchain(toolchain, osVariant)
            testToolchain(toolchain, osVariant)
            tagArtifacts(osVariant, autoMerge)
        }
    }
    return stage
}

/* Run pre merge */
def mainAction() {
    Boolean autoMerge = env.RUN_TYPE == 'merge' ? true : false
    String primaryBuilderImage = "${env.JFROG_CONTAINER_REGISTRY}/nextbuilder-${PRIMARY_RELEASE}:${env.NEXTBUILDER_IMAGE_TAG}"
    Builder toolchain = new Builder(ctx: this, projectName:'next-llvm-project')

    stage('Bump toolchain version') {
        println "==================== SETUP FROM HERE ON ===================="
        nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-bump-toolchain", primaryBuilderImage, "1", "16", "2", "32", CING_VERSION) {
            println "==================== SETUP FINISHED ===================="
            bumpToolchain(toolchain, env.RUN_TYPE, autoMerge, false)
        }
    }

    String[] os_variants = [ROCKY_RELEASE, DEBIAN12_RELEASE]
    Map llvmMap = [failFast: autoMerge]

    os_variants.each { osVariant ->
        String osVariantName = osVariant
        llvmMap[osVariantName] = prepareStage(toolchain, osVariantName, autoMerge)
    }

    // build and test toolchain on debian-12/rocky-9 in parallel
    parallel(llvmMap)

    if (currentBuild.currentResult == 'SUCCESS') {
        stage('Push bump toolchain version') {
            println "==================== SETUP FROM HERE ON ===================="
            nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-bump-toolchain",
                    primaryBuilderImage, "1", "16", "2", "32", CING_VERSION) {
                println "==================== SETUP FINISHED ===================="
                bumpToolchain(toolchain, env.RUN_TYPE, autoMerge, true)
            }
        }
    }
}

timestamps {
    ansiColor('xterm') {
        // validate if merge pipeline is triggered by user
        if (userBuildCause() && env.RUN_TYPE == 'merge') {
            error("Do not run pre-merge pipeline for merge manually")
        }
        // trigger pre-merge pipeline
        withEnv(["NEXT_HOME=/opt/nextsilicon"]) {
            mainAction()
        }
    }
}
