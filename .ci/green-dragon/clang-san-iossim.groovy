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
        incremental: false,
        env_vars: [
            "MACOSX_DEPLOYMENT_TARGET": "10.14"
        ]
    ],
    testConfig: [
        timeout: 90,
        env_vars: [
            "SANITIZER_IOSSIM_TEST_DEVICE_IDENTIFIER": "iPhone 15"
        ],
        custom_script: '''
            EXIT_CODE=0
            export COMPILER_RT_TEST_DIR="$WORKSPACE/clang-build/runtimes/runtimes-bins/compiler-rt/test"

            cd $COMPILER_RT_TEST_DIR/asan && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-asan-IOSSimX86_64Config.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/asan/IOSSimX86_64Config/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/tsan && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-tsan-IOSSimX86_64Config.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/tsan/IOSSimX86_64Config/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/ubsan && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-ubsan-AddressSanitizer-iossim-x86_64.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/ubsan/AddressSanitizer-iossim-x86_64/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/ubsan && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-ubsan-Standalone-iossim-x86_64.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/ubsan/Standalone-iossim-x86_64/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/ubsan && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-ubsan-ThreadSanitizer-iossim-x86_64.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/ubsan/ThreadSanitizer-iossim-x86_64/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/sanitizer_common && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-sanitizer_common-asan-iossim-x86_64.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/sanitizer_common/asan-x86_64-iossim/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/sanitizer_common && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-sanitizer_common-tsan-iossim-x86_64.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/sanitizer_common/tsan-x86_64-iossim/ || EXIT_CODE=1

            cd $COMPILER_RT_TEST_DIR/sanitizer_common && python3 $WORKSPACE/clang-build/./bin/llvm-lit \
              --xunit-xml-output=testresults-sanitizer_common-ubsan-iossim-x86_64.xunit.xml -v -vv --timeout=600 \
              $COMPILER_RT_TEST_DIR/sanitizer_common/ubsan-x86_64-iossim/ || EXIT_CODE=1

            exit $EXIT_CODE
        ''',
        junit_patterns: [
            "clang-build/**/testresults-*.xunit.xml"
        ]
    ]
)