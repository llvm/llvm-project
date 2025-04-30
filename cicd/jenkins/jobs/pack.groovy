#!/usr/bin/env groovy

pipelineJob('next-llvm-project-pack') {
  displayName('next-llvm-project-pack')
  description('''Build nextllvm sw package for multiple OS - debian-10, rhel-8
Default params values create a package from next_release_170 latest commit''')

  disabled(false)

  properties {
    buildDiscarder {
      strategy {
        logRotator {
          daysToKeepStr('30')
          numToKeepStr('20')
          artifactDaysToKeepStr('')
          artifactNumToKeepStr('')
        }
      }
    }

    parameters {
      parameterDefinitions {
        activeChoice {
          name('OS_VARIANTS')
          description('Select the OS variants')
          script {
            groovyScript {
              script {
                script('return ["debian-10:disabled","rhel-8:disabled","rocky-9:selected","debian-12:selected"]')
                sandbox(false)
              }
              fallbackScript {
                script('return ["true"]')
                sandbox(false)
              }
            }
          }
          choiceType('PT_CHECKBOX')
          filterLength(1)
          randomName('')
          filterable(false)
        }

        stringParam {
          name('BRANCH_NAME')
          defaultValue('next_release_170')
          description('branch used for checkout')
          trim(false)
        }

        stringParam {
          name('UTILS_BRANCH_NAME')
          defaultValue('master')
          description('used to checkout utils while running pull next_home tar code')
          trim(true)
        }

        stringParam {
          name('TOOLCHAIN_JOB')
          defaultValue('next_release_170')
          description('''Use next_release_170 or PR-XX in order to pull next_home tar and use it for packaging
Leave empty in order to build next-llvm-project from scratch using the BRANCH_NAME param''')
          trim(false)
        }

        stringParam {
          name('REPO_NAME')
          defaultValue('next-llvm-project')
          description('repo name to build')
          trim(false)
        }

        stringParam {
          name('SLACK_CHANNEL')
          defaultValue('sw-release-ci')
          description('slack channel')
          trim(false)
        }

        separator {
          name('PARAMS USED FOR RELEASE CRON')
          separatorStyle('')
          sectionHeader('PARAMS USED FOR RELEASE CRON')
          sectionHeaderStyle('')
          description('')
        }

        booleanParam {
          name('GIT_TAG')
          defaultValue(false)
          description('used in release cron in order to tag')
        }

        stringParam {
          name('VERSION')
          defaultValue('')
          description('used to pass a version into the release cron')
          trim(false)
        }

        choiceParam {
          name('T_TYPE')
          description('used in release cron')
          choices('''rc
release''')
        }

        separator {
          name('PARAMS USED FOR NIGHTLY CRON')
          sectionHeader('PARAMS USED FOR NIGHTLY CRON')
          separatorStyle('')
          sectionHeaderStyle('')
        }

        booleanParam {
          name('NIGHTLY')
          defaultValue(false)
          description('used to trigger nightly build and run cron')
        }

        stringParam {
          name('NEXTUTILS_BRANCH')
          defaultValue('master')
          description('used in nightly cron to pass to nextutils-pack job')
          trim(true)
        }

        stringParam {
          name('ARTIFACTORY_NEXTLLVM_BRANCH')
          defaultValue('next_release_170')
          description('used in nightly cron')
          trim(false)
        }
      }

      pipelineTriggers {
        triggers {
          parameterizedCron {
            parameterizedSpecification('''TZ=Israel
            00 19 * * * % NIGHTLY=True;SLACK_CHANNEL=nightly-master
            #H 19 * * * % NIGHTLY=True;SLACK_CHANNEL=nightly-master
            #05 00 * * 0 % GIT_TAG=True''')
          }
        }
      }
    }
  }

  throttleConcurrentBuilds {
    maxPerNode(0)
    maxTotal(2)
    throttleDisabled(false)
  }

  definition {
    cpsScm {
      lightweight(true)
      scm {
        git {
          branch('remotes/origin/${BRANCH_NAME}')
          remote {
            credentials('Jenkins-Next-user-github-API-token')
            url('https://github.com/nextsilicon/next-llvm-project.git')
          }
        }
      }
      scriptPath('cicd/jenkins/pipelines/pack.groovy')
    }
  }
}
