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
        cmake_type: 'Release',
        assertions: true,
        projects: 'clang;clang-tools-extra;lldb',
        runtimes: 'compiler-rt',
        timeout: 240,
        incremental: false,
        noinstall: true,
        env_vars: [
            "TMPDIR": '$WORKSPACE/tmpdir'
        ],
        pre_build_commands: 'rm -rf "$WORKSPACE/tmpdir" && mkdir -p "$WORKSPACE/tmpdir"',
        // Skip LLDB testing since it is done elsewhere.
        // Also skip most compiler-rt testing since upstream and the swiftlang fork are
        // mostly the same so doing lots of testing here would be redundant.
        cmake_flags: [
            "-DLLVM_TARGETS_TO_BUILD=X86;ARM;AArch64",
            "-DLLDB_ENABLE_SWIFT_SUPPORT=OFF",
            "-DLLDB_INCLUDE_TESTS=OFF",
            "-DLLDB_ENABLE_LZMA=OFF",
            "-DCOMPILER_RT_INCLUDE_TESTS=ON",
            "-DCOMPILER_RT_ENABLE_TEST_SUITES=builtins;bounds_safety",
            "-DCOMPILER_RT_BOUNDS_SAFETY_USE_LLDB=ON",
            "-DCOMPILER_RT_BOUNDS_SAFETY_USE_JUST_BUILT_LLDB=OFF"
        ]
    ],
    testConfig: [
        timeout: 240,
        env_vars: [
            "TMPDIR": '$WORKSPACE/tmpdir'
        ],
        // Run all checks with -k 0 so the job continues past individual failures and collects
        // all test results. The || true ensures Jenkins sees success so xunit results are
        // always archived; the test stage itself is marked unstable when failures are present.
        custom_script: '''
            ninja -v -k 0 -C "$WORKSPACE/clang-build" check-all || true
            rm -rf "$WORKSPACE/tmpdir"
        ''',
        junit_patterns: [
            "clang-build/**/testresults.xunit.xml"
        ]
    ]
)
