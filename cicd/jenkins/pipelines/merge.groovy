@Library('nextci@master') _

properties([[$class: 'JiraProjectProperty'],
  buildDiscarder(logRotator(artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '60',
    numToKeepStr: '30'))])

/* Return job type based on the comment that triggered the build */
@NonCPS
String getRunType() {

  def triggerCause = currentBuild.rawBuild.getCause(com.adobe.jenkins.github_pr_comment_build.GitHubPullRequestCommentCause)
  String runType = ''
  if (triggerCause) {
    String commentBody = triggerCause.getCommentBody()
    if (commentBody.contains('merge')) {
      runType = 'merge'
    } else if (commentBody.contains('test')) {
      runType = 'test'
    }
  }
  return runType
}

/* Return bump level based on the comment that triggered the build */
@NonCPS
String getBumpLevel() {

  def triggerCause = currentBuild.rawBuild.getCause(com.adobe.jenkins.github_pr_comment_build.GitHubPullRequestCommentCause)
  String bumpLevel = ''
  if (triggerCause) {
    String commentBody = triggerCause.getCommentBody()
    if (commentBody.contains('patch')) {
      bumpLevel = 'PATCHLEVEL'
    } else if (commentBody.contains('minor')) {
      bumpLevel = 'MINOR'
    } else if (commentBody.contains('major')) {
      bumpLevel = 'MAJOR'
    }
  }
  return bumpLevel
}

def mainAction(String bumpLevel, String runType) {

  Boolean autoMerge = runType == 'merge' ? true : false

  // validate PR is mergeable
  stage('Validate PR') {
    echo "Running pre-merge pipeline for ${runType}: source branch - ${pullRequest.headRef}, target branch - ${pullRequest.base}"
    if (pullRequest.mergeable) {
      echo "PR is mergeable, pre-merge pipeline will run"
    } else {
      if (autoMerge) {
        pullRequest.comment("Pre-merge failed. PR is not mergeable. See [build page](${currentBuild.absoluteUrl}) for more info")
        error("Pre-merge failed. PR is not mergeable. See [build page](${currentBuild.absoluteUrl}) for more info")
      } else {
        echo "PR is not mergeable, pre-merge pipeline for merge will fail"
      }
    }
  }

  // run pre-merge job
  stage('Run pre-merge') {
    status = build(propagate: false,
      job: 'toolchain-pre-merge', parameters: [
        string(name: 'TOOLCHAIN_TARGET', value: pullRequest.base),
        string(name: 'TOOLCHAIN_BRANCH', value: pullRequest.headRef),
        string(name: 'RUN_TYPE', value: runType),
        string(name: 'BUMP_TOOLCHAIN_LEVEL', value: bumpLevel),],
    )

    if (status.currentResult != 'SUCCESS') {
      pullRequest.comment("Pre-merge failed. See [build page](${status.absoluteUrl}) for more info")
      error("Pre-merge failed. See [build page](${status.absoluteUrl}) for more info")
    }
  }

  if (!autoMerge) {
    stage('Test pre-merge') {
      pullRequest.comment("Pre-merge test passed. See [build page](${status.absoluteUrl}) for more info")
      echo "Pre-merge test passed, exiting..."
    }
  } else {
    // validate PR is mergeable
    stage('Merge') {
      echo "Pre-merge test passed, merging PR"
      /*** The value of the mergeable attribute can be true, false, or null. If the value is null, then GitHub has started
      * a background job to compute the mergeability. After giving the job time to complete, resubmit the request. When the
      * job finishes, you will see a non-null value for the mergeable attribute in the response. If mergeable is true, then
      * merge_commit_sha will be the SHA of the test merge commit.
      *
      * https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#get-a-pull-request
      ***/
      echo "First, let's sleep to let the status checks update on GitHub."
      sleep (time: 10)

      echo "Now let's see the PR status"
      if (!pullRequest.mergeable) {
        //state ${pullRequest.mergeableState} will be used when plugin is upgraded.
        echo "mergeable is ${pullRequest.mergeable}, sleeping a bit more."
        sleep (time: 20)
        if (!pullRequest.mergeable) {
          //state ${pullRequest.mergeableState} will be used when plugin is upgraded.
          echo "Still ${pullRequest.mergeable}, sleeping for one last time"
          sleep (time: 30)
        }
      }

      if (!pullRequest.mergeable) {
        pullRequest.comment("Pre-merge failed. PR is not mergeable. See [build page](${currentBuild.absoluteUrl}) for more info")
        error("Pre-merge failed. PR is not mergeable. See [build page](${currentBuild.absoluteUrl}) for more info")
      }
      // merge PR
      retry (3) {
        echo "Trying to merge"
        pullRequest.merge(mergeMethod: "rebase")
        sleep (time: 10)
      }
    }
  }
}

timestamps {
  ansiColor('xterm') {
    String runType = getRunType()
    String bumpLevel = getBumpLevel()

    if (runType == '') {
      error("Run type is not set")
    }

    if (bumpLevel == '') {
      error("Bump level is not set")
    }

    if (runType == 'merge') {
      lock('toolchain-merge') {
        mainAction(bumpLevel, runType)
      }
    } else {
      mainAction(bumpLevel, runType)
    }
  }
}
