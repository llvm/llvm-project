#!/usr/bin/env groovy

multibranchPipelineJob('next-llvm-project-merge') {
  displayName('next-llvm-project-merge')
  description('''This multibranch pipeline listens for `[ci-...merge]` comments and triggers `toolchain-pre-merge` job.

When that succeeds, it then merges the PR.''')

  properties {
    pipelineTriggerProperty {
      createActionJobsToTrigger("HookPipelineCreation")
      deleteActionJobsToTrigger("HookPipelineDeleted")
    }
  }

  branchSources {
    branchSource {
      source {
        github {
          id('next-llvm-project-merge')
          repoOwner('nextsilicon')
          repository('next-llvm-project')
          repositoryUrl('https://github.com/nextsilicon/next-llvm-project.git')
          configuredByUrl(true)
          credentialsId('Jenkins-Next-user-github-API-token')

          traits {
            gitHubForkDiscovery {
              strategyId(0)
              trust {
                gitHubTrustPermissions()
              }
            }

            gitHubPullRequestDiscovery {
              strategyId(2)
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

            cleanBeforeCheckout {
              extension {
                deleteUntrackedNestedRepositories(false)
              }
            }

            gitHubNotificationContextTrait {
              contextLabel('end-to-end/pre-merge')
              typeSuffix(false)
            }
          }
        }

        strategy {
          allBranchesSame {
            props {
              suppressAutomaticTriggering {
                strategy("INDEXING")

                triggeredBranchesRegex('^$')
              }
              triggerPRCommentBranchProperty {
                commentBody('\\[CI-(MERGE|MERGE-PATCH|MERGE-MINOR|MERGE-MAJOR|TEST-PATCH|TEST-MINOR|TEST-MAJOR)\\]')
              }
            }
          }
        }
      }
    }
  }

  factory {
    workflowBranchProjectFactory {
      scriptPath('cicd/jenkins/pipelines/merge.groovy')
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
