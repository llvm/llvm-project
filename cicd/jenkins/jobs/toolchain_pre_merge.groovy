#!/usr/bin/env groovy

pipelineJob('toolchain-pre-merge') {
  displayName('toolchain-pre-merge')
  description('Toolchain pre merge pipeline')

  properties {
    buildDiscarder {
      strategy {
        logRotator {
          daysToKeepStr('60')
          numToKeepStr('30')
          artifactDaysToKeepStr('')
          artifactNumToKeepStr('')
        }
      }
    }

    parameters {
      parameterDefinitions {
        stringParam {
          name('TOOLCHAIN_BRANCH')
          description('Toolchain source branch')
        }

        stringParam {
          name('TOOLCHAIN_TARGET')
          description('Toolchain target branch')
        }

        choiceParam {
          name('BUMP_TOOLCHAIN_LEVEL')
          description('''Set bump toolchain version update type - X.Y.Z
X - MAJOR version update
Y - MINOR version update
Z - PATCHLEVEL version update''')
          choices('''PATCHLEVEL
MINOR
MAJOR''')
        }

        choiceParam {
          name('RUN_TYPE')
          choices('''test
merge''')
        }
      }
    }
  }

  definition {
    cpsScm {
      lightweight(true)
      scm {
        git {
          branch('remotes/origin/next_release_170')
          remote {
            credentials('Jenkins-Next-user-github-API-token')
            url('https://github.com/nextsilicon/next-llvm-project.git')
          }
          extensions {
            cloneOptions {
              depth(1)
              shallow(true)
              honorRefspec(true)
              noTags(true)
              reference('/repos/next-llvm-project.git')
              timeout(10)
            }

            checkoutOption {
              timeout(10)
            }

            submodule {
              disableSubmodules(true)
            }
          }
        }
      }

      scriptPath('cicd/jenkins/pipelines/toolchain_pre_merge.groovy')
    }
  }
}
