branchName = 'main'

properties([
    buildDiscarder(logRotator(numToKeepStr: '20')),
    disableConcurrentBuilds()
])

library identifier: "zorg-shared-lib@${branchName}",
        retriever: modernSCM([
            $class: 'GitSCMSource',
            remote: "https://github.com/llvm/llvm-zorg.git",
            credentialsId: scm.userRemoteConfigs[0].credentialsId
        ])

relay.pipeline([
    "llvm.org/clang-stage2-cmake-RgSan",
    "llvm.org/clang-stage2-cmake-RgTSan"
])