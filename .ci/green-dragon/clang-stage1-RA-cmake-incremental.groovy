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
        cmake_type: 'default',
        assertions: true,
        projects: 'clang',
        timeout: 150,
        incremental: true
    ],
    testConfig: [
        test_type: 'testlong',
        timeout: 150,
        junit_patterns: [
            "clang-build/**/testresults.xunit.xml"
        ]
    ]
)

