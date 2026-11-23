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
    defaultLabel: 'macos-arm64',
    buildConfig: [
        stage: 1,
        build_type: 'cmake',
        cmake_type: 'RelWithDebInfo',
        assertions: true,
        projects: 'clang;clang-tools-extra',
        runtimes: 'compiler-rt',
        timeout: 120,
        incremental: false,
        cmake_flags: [
            "-DLLVM_TARGETS_TO_BUILD=AArch64"
        ]
    ],
    testConfig: [
        test_type: 'testlong',
        timeout: 120,
        junit_patterns: [
            "clang-build/**/testresults.xunit.xml"
        ]
    ],
)
