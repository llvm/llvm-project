branchName = 'main'

library identifier: "zorg-shared-lib@${branchName}",
        retriever: modernSCM([
            $class: 'GitSCMSource',
            remote: "https://github.com/llvm/llvm-zorg.git",
            credentialsId: scm.userRemoteConfigs[0].credentialsId
        ])

clangPipeline([
    jobName: env.JOB_NAME,
    zorgBranch: branchName,
    buildConfig: [
        build_type: "clang",
        cmake_type: "RelWithDebInfo",
        thinlto: true,
        projects: "clang",
        runtimes: "libunwind;compiler-rt",
        stage: 2,
        timeout: 1200,
        incremental: false,
        stage1Job: 'clang-stage1-RA',
        cmake_flags: [
            "-DCMAKE_DSYMUTIL=\${WORKSPACE}/host-compiler/bin/dsymutil"
        ]
    ],
    testConfig: [
        test_command: "clang",
        test_type: "test",
        timeout: 420,
        junit_patterns: [
            "clang-build/**/testresults.xunit.xml"
        ]
    ]
])