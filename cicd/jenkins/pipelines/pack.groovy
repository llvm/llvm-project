@Library('nextci@master') _

import org.nextsilicon.Builder
import org.nextsilicon.NextSlack
import groovy.transform.Field

/* This drives next-llvm-project-pack pipeline. */

/*Define global parameters*/
@Field String NEXT_HOME = "/opt/linux_next_home"

// cing configuration
@Field String LIMIT_OUTPUT = "1000"
@Field String GLOBAL_REPO_NAME = "next-llvm-project"
@Field String MAIN_BRANCH = "next_release_170"
@Field String CING_VERSION = "main"

@Field String DEBIAN12_RELEASE = "debian-12"
@Field String ROCKY_RELEASE = "rocky-9"
@Field String PRIMARY_RELEASE = ROCKY_RELEASE

void buildOsVariant(NextSlack slackReport, String osVariant) {
  return {
    osImage = "${env.JFROG_CONTAINER_REGISTRY}/nextbuilder-${osVariant}:${env.NEXTBUILDER_IMAGE_TAG}"
    println "==================== SETUP FROM HERE ON ===================="
    nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-${osVariant}", osImage, '16', '32', '', '', CING_VERSION) {
      println "==================== SETUP FINISHED ===================="
      slackReport.stageWrapper('Env setup', ['os variant': osVariant]) {
        echo "OS_VARIANT=${osVariant}"
        echo "NODE_NAME=${NODE_NAME}"
      }
      Builder builder = new Builder(ctx: this, projectName: params.REPO_NAME)
      sshagent(['github-ssh-key']) {
        sh(script: "mkdir -p ~/.ssh/",
        label: 'mkdir -p ~/.ssh/')
        sh(script: "ssh-keyscan -H github.com > ~/.ssh/known_hosts",
        label: 'set ssh known hosts')
        String checkoutBranch = "${params.BRANCH_NAME}"
        // If NIGHTLY param is enabled (starts the nightly cron), a git commit hash
        // is stored at this point for each repo and used in the entire process -
        // git hashes are passed through the cron pipeline and are used in checkout
        // in the relvevant repo. This check is done only for rhel os.
        if (params.NIGHTLY) {
          checkoutBranch = sh(script: "git ls-remote git@github.com:nextsilicon/next-llvm-project.git HEAD | cut -f1", returnStdout: true).trim()
          if (osVariant == PRIMARY_RELEASE) {
            llvmGitHash = sh(script: "git ls-remote git@github.com:nextsilicon/next-llvm-project.git HEAD | cut -f1", returnStdout: true).trim()
            utilsGitHash = sh(script: "git ls-remote git@github.com:nextsilicon/nextutils.git HEAD | cut -f1", returnStdout: true).trim()
            echo "llvmGitHash: ${llvmGitHash}"
            echo "utilsGitHash: ${utilsGitHash}"
          }
        }
        slackReport.stageWrapper('Checkout', ['os variant': osVariant]) {
          dir(params.REPO_NAME) {
            retry(3) {
              builder.checkoutRepo(checkoutBranch)
            }
          }
        }
        if (params.TOOLCHAIN_JOB != '') {
          slackReport.stageWrapper('Pull next home tar from ' + params.TOOLCHAIN_JOB, ['os variant': osVariant]) {
            pullToolchain()
          }
        } else {
          slackReport.stageWrapper('Build', ['os variant': osVariant]) {
            dir(params.REPO_NAME) {
              withEnv(["NEXT_HOME=${NEXT_HOME}"]) {
                sh(script: "cing-builder -limit-output ${LIMIT_OUTPUT} ${GLOBAL_REPO_NAME} -clean --compile-type --release -install -build-libs -jobs-number ${CORE_NUMBER}", label: "Cing exec builder")
              }
            }
          }
        }

        if (params.BRANCH_NAME != "${MAIN_BRANCH}") {
          slackReport.stageWrapper('tar and Upload NEXHOME tar ', ['os variant': osVariant]) {
            sh(script: """
              cing-artifactory -limit-output ${LIMIT_OUTPUT} upload \
              -src-artifact ${NEXT_HOME} \
              -dst-artifact generic-repo/nextsilicon-files/nextllvm/${osVariant}/${params.BRANCH_NAME}/linux_next_home.tar.xz-${MAIN_BRANCH}
            """ ,label: "Cing exec artifactory upload")
          }
        }
        slackReport.stageWrapper('Packaging toolchain', ['os variant': osVariant]) {
          dir(params.REPO_NAME) {
            withEnv(["NEXT_HOME=${NEXT_HOME}", \
                      "BRANCH_NAME=${params.BRANCH_NAME}", \
                      "REPO_NAME=${params.REPO_NAME}"]) {
              // build OS variant package
              sh(script: "./packages/build_release.sh --pack-release --os-variant ${osVariant}",
              label: "Build next-llvm-project packages for ${osVariant}")
            }
          }
        }
        slackReport.stageWrapper('Update remote artifactory', ['os variant': osVariant]) {
          dir(params.REPO_NAME) {
            withCredentials([usernamePassword(credentialsId: 'jenext', \
                              usernameVariable: 'USERNAME', \
                              passwordVariable: 'PASSWORD')]) {
              sh(script: """
                  ./packages/build_release.sh --update-artifactory --os-variant "${osVariant}"
                  """,
                  label: 'Upload packages to remote artifactory')
            }
          }
        }
      }
    }
  }
}

void pullToolchain() {
  Builder utils = new Builder(ctx: this, projectName:'nextutils')
  String utilsPath = "${env.WORKSPACE}/${utils.projectName}"
  dir(utilsPath) {
    utils.checkoutRepo('master')
    // While using nightly cron, get llvm version and pass it
    // in the pipe to use later for pulling same tar home version
    // while building and packaging nextutils
    withEnv(["NEXT_HOME=${NEXT_HOME}"]) {
      if (params.NIGHTLY) {
        LLVM_VERSION = sh(script: "cmake -P '${utilsPath}/cmake/PrintLLVM.txt' | cut -c 4-", returnStdout: true).trim()
        echo "LLVM_VERSION: ${LLVM_VERSION}"
      }
      if (params.TOOLCHAIN_JOB == "${MAIN_BRANCH}") {
        sh(script: "cing-builder-setup -limit-output ${LIMIT_OUTPUT} nextutils --setup --fetch-next-toolchain", label: "Cing exec builder-setup ${MAIN_BRANCH} toolchain")
      } else if (params.TOOLCHAIN_JOB =~ 'PR') {
        sh(script: "cing-builder-setup -limit-output ${LIMIT_OUTPUT} nextutils --setup --fetch-next-toolchain -toolchain-job ${params.TOOLCHAIN_JOB}", label: "Cing exec builder-setup ${params.TOOLCHAIN_JOB} toolchain")

      } else {
        error('TOOLCHAIN_JOB value is not correct')
      }
    }
  }
}

String slackChan = params.SLACK_CHANNEL ? params.SLACK_CHANNEL : 'sw-release-ci'
@Field String llvmGitHash = ""
@Field String utilsGitHash = ""
NextSlack slackReport = new NextSlack(ctx: this, channel: slackChan,\
    skipNotificationOnSuccess : true , iconEmoji: ':nextllvm:')



timestamps {
  ansiColor('xterm') {
    slackReport.jobWrapper {

      List osVariants = []
      osVariants = params.OS_VARIANTS ? params.OS_VARIANTS.split(',') : error("Os not Provided")
      String primaryBuilderImage = "${env.JFROG_CONTAINER_REGISTRY}/nextbuilder-${PRIMARY_RELEASE}:${env.NEXTBUILDER_IMAGE_TAG}"
      println "==================== SETUP FROM HERE ON ===================="
      nextK8s.kubeWrapperSingleContainer("${BUILD_TAG}-main", primaryBuilderImage, "1", "1", "1", "8", CING_VERSION) {
        println "==================== SETUP FINISHED ===================="
        Map tasks = [failFast: false]
        osVariants.each { osVariant ->
          String osVariantName = osVariant
          tasks[osVariantName] = buildOsVariant(slackReport, osVariantName)
        }
        parallel(tasks)
      }
      // Trigger a new job if this job was successful and the GIT_TAG param
      // is enabled. This process eventually creates a tar file and deploypackages
      // starts using the tar. Mainly used when creating release or rc branches.
      if (currentBuild.currentResult == 'SUCCESS' && params.GIT_TAG) {

        Map stageInfo = [:]
        slackReport.stageWrapper('nextutils-pack', stageInfo) {
        status = build(propagate: false, wait: false,
          job: 'nextutils-pack',
          parameters: [
            string(name: 'Description', value: "Auto-trigger by: ${env.JOB_NAME}-${currentBuild.displayName}"),
            [$class: 'BooleanParameterValue', name: 'GIT_TAG', value: true],
            [$class: 'BooleanParameterValue', name: 'USE_BACKUP', value: true],
            string(name: 'NEXTLLVM_BUILD_ID', value: "${env.BUILD_ID}"),
            string(name: 'T_TYPE', value: "${params.T_TYPE}"),
            string(name: 'BRANCH_NAME', value: "${params.UTILS_BRANCH_NAME}"),
            string(name: 'TOOLCHAIN_BRANCH', value: "${params.BRANCH_NAME}"),
            string(name: 'VERSION', value: "${params.VERSION}"),
            string(name: 'SLACK_CHANNEL', value: params.SLACK_CHANNEL),
          ])
        }
      }
      // Trigger a new job if this job was successful and the NIGHTLY param
      // is enabled. Params are passed to next jobs, until eventually deploypackaes
      // job is running using the packages generated in the entire NIGHTLY pipeline
      if (currentBuild.currentResult == 'SUCCESS' && params.NIGHTLY) {
        Map stageInfo = [:]
        slackReport.stageWrapper('nextutils-pack', stageInfo) {
        status = build(propagate: false, wait: false,
          job: 'nextutils-pack',
          parameters: [
            string(name: 'COMPILE_TYPE', value: "RelWithDebInfo"),
            [$class: 'BooleanParameterValue', name: 'GIT_TAG', value: true],
            string(name: 'T_TYPE', value: "rc"),
            [$class: 'BooleanParameterValue', name: 'NIGHTLY', value: true],
            string(name: 'Description', value: "Nightly triggered by: ${env.JOB_NAME}-${currentBuild.displayName}"),
            string(name: 'ARTIFACTORY_NEXTLLVM_BRANCH', value: "${params.BRANCH_NAME}"),
            string(name: 'NEXTUTILS_BRANCH', value: "${params.NEXTUTILS_BRANCH}"),
            string(name: 'UTILS_HASH', value: "${utilsGitHash}"),
            string(name: 'LLVM_HASH', value: "${llvmGitHash}"),
            string(name: 'LLVM_VERSION', value: "${LLVM_VERSION}"),
            string(name: 'NEXTLLVM_BUILD_ID', value: "${env.BUILD_ID}"),
            string(name: 'SLACK_CHANNEL', value: "nightly-master"),
          ])
        }
      }
    }
  }
}
