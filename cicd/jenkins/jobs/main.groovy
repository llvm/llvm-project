#!/usr/bin/env groovy

multibranchPipelineJob('next-llvm-project') {
  displayName('next-llvm-project')
  description('LLVM building')

  branchSources {
    branchSource {
      source {
        github {
          id('next-llvm-project')
          repoOwner('nextsilicon')
          repository('next-llvm-project')
          repositoryUrl('https://github.com/nextsilicon/next-llvm-project.git')
          configuredByUrl(true)
          credentialsId('Jenkins-Next-user-github-API-token')

          traits {
            gitHubBranchDiscovery {
              strategyId(0)
            }

            gitHubForkDiscovery {
              strategyId(0)
              trust {
                gitHubTrustPermissions()
              }
            }

            gitHubPullRequestDiscovery {
              strategyId(2)
            }

            headWildcardFilter {
              includes('PR-* next_release_90 next_release_120 next_release_140 next_release_160 next_release_170 llvm_main')
              excludes("")
            }

            cloneOption {
              extension {
                noTags(false)
                honorRefspec(true)
                reference('$ROOT_FOLDER/next-llvm-project.git')
                shallow(false)
                timeout(10)
              }
            }

            submoduleOption {
              extension {
                recursiveSubmodules(true)
                parentCredentials(true)
                shallow(true)
                depth(1)
              }
            }

            localBranch()

            gitHubSshCheckout {
              credentialsId('github-ssh-key')
            }

            refSpecs {
              templates {
                refSpecTemplate {
                  value('+refs/heads/*:refs/remotes/@{remote}/*')
                }
              }
            }
          }
        }
      }
    }
  }

  factory {
    workflowBranchProjectFactory {
      scriptPath('cicd/jenkins/pipelines/main.groovy')
    }
  }

  triggers {
    periodicFolderTrigger {
      interval('30m')
    }
  }

  orphanedItemStrategy {
    discardOldItems {
      daysToKeep(14)
    }
  }
}
