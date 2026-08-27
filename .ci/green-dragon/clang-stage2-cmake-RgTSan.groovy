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
    stages: ['checkout', 'fetch', 'build'],
    buildConfig: [
        stage: 2,
        stage1Job: 'clang-stage1-RA',
        build_type: 'cmake',
        cmake_type: 'RelWithDebInfo',
        projects: 'clang;clang-tools-extra',
        runtimes: 'libcxx;libcxxabi;compiler-rt',
        cmake_build_target: 'LTO',
        noinstall: true,
        thinlto: true,
        sanitizer: 'Thread',
        incremental: false,
        cmake_flags: [
            "-DLLVM_BUILD_RUNTIME=OFF"
        ],
        env_vars: [
            "DYLD_LIBRARY_PATH": "\$WORKSPACE/host-compiler/lib/"
        ]
    ]
)
