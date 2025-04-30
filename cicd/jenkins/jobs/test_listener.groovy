#!/usr/bin/env groovy

multibranchPipelineJob('next-llvm-test-listener') {
  displayName('next-llvm-test-listener')
  description('This multibranch pipeline waits for PR comments of `[ci-...` style and triggers other jobs for testing the PRs.')

  branchSources {
    branchSource {
      source {
        github {
          id('next-llvm-test-listener')
          repoOwner('nextsilicon')
          repository('next-llvm-project')
          repositoryUrl('https://github.com/nextsilicon/next-llvm-project.git')
          configuredByUrl(true)
          credentialsId('github-app-jenkins')

          traits {
            gitHubPullRequestDiscovery {
              strategyId(2)
            }

            headWildcardFilter {
              includes('PR-*')
              excludes('')
            }

            gitHubStatusChecks {
              name('Tests')
              skip(true)
              skipNotifications(true)
              skipProgressUpdates(true)
              suppressLogs(false)
              unstableBuildNeutral(false)
            }

            cloneOption {
              extension {
                noTags(false)
                honorRefspec(true)
                reference('$ROOT_FOLDER/next-llvm-project.git')
                shallow(false)
                timeout(1)
              }
            }

            submoduleOption {
              extension {
                disableSubmodules(true)
                recursiveSubmodules(true)
                parentCredentials(true)
                shallow(true)
                depth(1)
                reference('$ROOT_FOLDER/next-llvm-project.git')
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

        strategy {
          allBranchesSame {
            props {
              suppressAutomaticTriggering {
                strategy("NONE")

                triggeredBranchesRegex('^$')
              }
              triggerPRCommentBranchProperty {
                commentBody('\\[ci-.*')
                allowUntrusted(true)
              }
            }
          }
        }
      }
    }
  }

  factory {
    workflowBranchProjectFactory {
      scriptPath('cicd/jenkins/pipelines/test_listener.groovy')
    }
  }

  triggers {
    periodicFolderTrigger {
      interval('7d')
    }
  }

  orphanedItemStrategy {
    discardOldItems {
      daysToKeep(14)
    }
  }
}
