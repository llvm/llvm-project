branchName = 'main'

library identifier: "zorg-shared-lib@${branchName}",
        retriever: modernSCM([
            $class: 'GitSCMSource',
            remote: "https://github.com/llvm/llvm-zorg.git",
            credentialsId: scm.userRemoteConfigs[0].credentialsId
        ])

clangPipeline(
    jobName: env.JOB_NAME,
    zorgBranch: branchName,
    buildConfig: [
        stage: 1,
        build_type: 'cmake',
        cmake_type: 'RelWithDebInfo',
        assertions: true,
        projects: 'clang;clang-tools-extra',
        runtimes: 'compiler-rt',
        timeout: 120,
        incremental: false
    ],
    testConfig: [
        test_type: 'testlong',
        timeout: 120,
        junit_patterns: [
            "clang-build/**/testresults.xunit.xml"
        ]
    ],
    triggeredJobs: [
        'llvm.org/clang-stage2-cmake-RgSan_relay',
        'llvm.org/clang-stage2-Rthinlto_relay',
        'llvm.org/relay-lnt-ctmark',
        'llvm.org/relay-test-suite-verify-machineinstrs'
    ]
)
