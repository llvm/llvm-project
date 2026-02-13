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
        stage: 2,
        stage1Job: 'clang-stage1-RA',
        build_type: 'cmake',
        cmake_type: 'RelWithDebInfo',
        projects: 'clang;clang-tools-extra',
        test_timeout: 2400, // individual test timeout in seconds
        timeout: 1440, // build stage timeout in minutes
        incremental: false,
        sanitizer: 'Address;Undefined',
        cmake_flags: [
            "-DLIBCXX_INCLUDE_TESTS=OFF"
        ],
        env_vars: [
            "HOST_INC_DIR": "\$WORKSPACE/host-compiler/bin/../include"
        ]
    ],
    testConfig: [
        test_command: 'cmake',
        test_type: 'test',
        test_targets: [
            'check-llvm',
            'check-clang'
        ],
        timeout: 1440, // test stage timeout in minutes
        env_vars: [
            "ASAN_SYMBOLIZER_PATH": "\$WORKSPACE/host-compiler/bin/llvm-symbolizer"
        ],
        junit_patterns: [
            "clang-build/**/testresults.xunit.xml"
        ]
    ]
)
