@Library('nextci@master') _

import groovy.transform.Field

@Field def SMOKETESTS_BUILD_NUMBER = null

@Field def VARIETY_SMOKETESTS    = 'Smoketests'
@Field def VARIETY_E2E_SIM       = "E2E Sim"
@Field def VARIETY_E2E_HW_SIM    = "E2E HwSim"
@Field def VARIETY_E2E_SI        = "E2E Si"
@Field def VARIETY_E2E_4H        = "E2E 4h"
@Field def VARIETY_E2E_24H       = "E2E 24H"


@Field tests_results = []

properties([[$class: 'JiraProjectProperty'],
    buildDiscarder(logRotator(artifactDaysToKeepStr: '',
        artifactNumToKeepStr: '',
        daysToKeepStr: '30',
        numToKeepStr: '150'))])

// Get the comment body from the trigger cause
@NonCPS
String getCommentBody() {
    String commentBody = ""
    def triggerCause = currentBuild.rawBuild.getCause(
            com.adobe.jenkins.github_pr_comment_build.GitHubPullRequestCommentCause)
    if (!triggerCause) { return "" }
    return triggerCause.getCommentBody()
}

@NonCPS
def getRequiredTests(String body) {
    println "Processing comment: `${body}`"
    def ret_value = []

    def regex = /(?i)\[ci-([A-Za-z0-9_-]+)\]/
    def matcher = ( body =~ regex )
    if (matcher.find()) {
        matcher.each { this_match ->
            def test_spec = this_match[1]
            ret_value += test_spec.toLowerCase()
        }
    }
    matcher = null
    return ret_value
}

def publishCheck(String state, String variety, String title="", String conclusion="SUCCESS") {
    println "Inside doPublish..."

    if (!title) {
        def stateToMessage = [
            "IN_PROGRESS": "Running now...",
            "QUEUED": "Expected to run",
            "NONE": "None",
            "COMPLETED": "Completed",
        ]
        title = stateToMessage[state]
    }

    publishChecks detailsURL: env.BUILD_URL,
        name: variety,
        status: state, // IN_PROGRESS / QUEUED / NONE / COMPLETED
        conclusion: conclusion, // 'SUCCESS'[default]/'NEUTRAL'/FAILURE/TIME_OUT/CANCELED/ SKIPPED
        title: title
}


// Run the tests for the given job name
def runOneTestVariety(String jobTitle, String jobName, List parameters) {
    stage("run ${jobName}") {
        publishCheck('IN_PROGRESS', jobTitle)

        def status = build job: jobName,
            propagate: false,
            parameters: parameters

        SMOKETESTS_BUILD_NUMBER = status.number

        if (status.currentResult != "SUCCESS") {
            publishCheck('COMPLETED', jobTitle, "", 'FAILURE')
            tests_results += ":eyes: `${jobTitle}` tests failed: [${jobName}-${status.number}](${status.absoluteUrl})."
            error("${jobName} failed")
        }

        publishCheck('COMPLETED', jobTitle)
        tests_results += ":clap: `${jobTitle}` tests passed: [${jobName}-${status.number}](${status.absoluteUrl})."
        echo "${jobName} passed"
    }
}

def getCommonParameters() {
    def reason = "nextllvm/${env.BRANCH_NAME} `${commentBody}` run"

    List common_parameters = [
        string(name: 'UTILS_BRANCH', value: "master"),
        string(name: 'TOOLCHAIN_TAG', value: env.BRANCH_NAME),
        booleanParam(name: 'NO_TOOLCHAIN_CHECK', value: true),
        string(name: 'Description', value: reason),
    ]
    return common_parameters
}

def run4h() {
    // announce expected check
    publishCheck('QUEUED', VARIETY_E2E_4H)
    // trigger CI-G2-OnGuest with 4h run
    def job_name = "CI-G2-OnGuest"
    List parameters = getCommonParameters()
    parameters << string(name: 'Pipeline', value: "default_collect")
    parameters << string(name: 'AppsFromFile', value: "applications/automatic_runs/sw-gen-2-apps-ci-4h.txt")
    parameters << string(name: 'COMPILE_TYPE', value: "relwithdebuginfo")
    parameters << booleanParam(name: 'STORE_MILLS', value: true)
    parameters << string(name: 'DISK_SIZE', value: "80")
    runOneTestVariety(VARIETY_E2E_4H, job_name, parameters)
}

def run24h() {
    // announce expected check
    publishCheck('QUEUED', VARIETY_E2E_24H)

    // trigger CI-G2-OnGuest with nightly run for 24h
    def job_name = "CI-G2-OnGuest"
    List parameters = getCommonParameters()
    parameters << string(name: 'Pipeline', value: "default_collect")
    parameters << string(name: 'AppsFromFile', value: "applications/automatic_runs/sw-gen-2-apps-ci-nightly.txt")
    parameters << string(name: 'COMPILE_TYPE', value: "relwithdebuginfo")
    parameters << booleanParam(name: 'STORE_MILLS', value: true)
    parameters << string(name: 'DISK_SIZE', value: "80")
    parameters << string(name: 'SUBMODULES_LIST', value: "rdma-core,ompi,ucx")
    parameters << string(name: 'JOB_TIMEOUT', value: "240")
    runOneTestVariety(VARIETY_E2E_24H, job_name, parameters)
}

def runSmoketests() {
    // announce expected checks
    publishCheck('QUEUED', VARIETY_SMOKETESTS)
    publishCheck('QUEUED', VARIETY_E2E_SIM, "Will run after ${VARIETY_SMOKETESTS}")
    publishCheck('QUEUED', VARIETY_E2E_HW_SIM, "Will run after ${VARIETY_SMOKETESTS}")
    publishCheck('QUEUED', VARIETY_E2E_SI, "Will run after ${VARIETY_SMOKETESTS}")

    // trigger smoke test
    List smk_p = getCommonParameters()
    smk_p << string(name: 'UTILS_PR_TARGET', value: "")
    smk_p << string(name: 'EMAIL_WATCHER_LIST', value: '')
    smk_p << booleanParam(name: 'email', value: true)
    smk_p << booleanParam(name: 'SKIP_POST_MERGE', value: true)

    def job_name = "nextutils-smoke-test"
    runOneTestVariety(VARIETY_SMOKETESTS, job_name, smk_p)

    def job_package_source = "${job_name}-${SMOKETESTS_BUILD_NUMBER}"

    def parallels = [
        "E2E Sim": {
            List e2e_s = getCommonParameters()
            e2e_s << string(name: 'JOB_PACKAGE', value: job_package_source)
            e2e_s << string(name: 'AppsFromFile', value: "applications/automatic_runs/e2e.txt")
            e2e_s << string(name: 'COMPILE_TYPE', value: "relwithdebuginfo")
            e2e_s << booleanParam(name: 'STORE_MILLS', value: true)
            e2e_s << string(name: 'SUBMODULES_LIST', value: "rdma-core,ompi,ucx")
            runOneTestVariety(VARIETY_E2E_SIM, "Gen2/NextimatorCI", e2e_s)
        },
        "E2E HwSim": {
            List e2e_hws = getCommonParameters()
            e2e_hws << string(name: 'JOB_PACKAGE', value: job_package_source)
            e2e_hws << string(name: 'AppsFromFile',
                value: "applications/automatic_runs/sw-gen-2-cm-libcalls-hwsim-post-merge.txt")
            e2e_hws << string(name: 'COMPILE_TYPE', value: "relwithdebuginfo")
            e2e_hws << booleanParam(name: 'STORE_MILLS', value: true)
            e2e_hws << string(name: 'SUBMODULES_LIST', value: "rdma-core,ompi,ucx")
            runOneTestVariety(VARIETY_E2E_HW_SIM, "Gen2/NextimatorCI", e2e_hws)
        },
        "E2E Si": {
            List e2e_si = getCommonParameters()
            e2e_si << string(name: 'JOB_PACKAGE', value: job_package_source)
            e2e_si << string(name: 'Pipeline', value: "default_collect")
            e2e_si << string(name: 'AppsFromFile', value: "applications/automatic_runs/sw-gen-2-apps-ci-post-merge.txt")
            e2e_si << string(name: 'COMPILE_TYPE', value: "relwithdebuginfo")
            e2e_si << booleanParam(name: 'STORE_MILLS', value: true)
            e2e_si << string(name: 'DISK_SIZE', value: "80")
            job_name = "CI-G2-OnGuest"
            runOneTestVariety(VARIETY_E2E_SI, job_name, e2e_si)
        }
    ]
    parallel parallels
}

String supportedTriggers = """Supported trigger comments:
                    |- `[ci-help]` - display this help comment
                    |- `[ci-smoketests]` - trigger [nextutils-smoke-test](https://jenkins.k8s.nextsilicon.com/job/nextutils-smoke-test) and then three E2E jobs
                    |- `[ci-e2e-4h]` - trigger [CI-G2-OnGuest](https://jenkins.k8s.nextsilicon.com/job/CI-G2-OnGuest/) configured as the SW E2E Gen2 CI 4h run.
                    |- `[ci-e2e-24h]` - trigger [CI-G2-OnGuest](https://jenkins.k8s.nextsilicon.com/job/CI-G2-OnGuest/) configured as the SW E2E Gen2 CI nightly run.
""".stripMargin()

String commentBody = getCommentBody()

//@NonCPS
//def printParams() {
//    env.getEnvironment().each { name, value -> println "Name: $name -> Value $value" }
//}
//printParams()

timestamps {
ansiColor('xterm') {
try {
    def tests = getRequiredTests(commentBody)

    for (String one_test : tests) {
        if (one_test == 'help') { // [ci-help]
            pullRequest.comment(supportedTriggers)
            continue
        }
        if (one_test == 'smoketests') {
            runSmoketests()
            continue
        }
        if (one_test == 'e2e-4h') {
            run4h()
            continue
        }
        if (one_test == 'e2e-24h') {
            run24h()
            continue
        }
    }

} finally {
    if (tests_results) {
        pullRequest.comment("Tests results:\n" + tests_results.join('\n'))
    }
}
}
}
