branchName = 'main'

properties([
    disableConcurrentBuilds()
])

library identifier: "zorg-shared-lib@${branchName}",
        retriever: modernSCM([
            $class: 'GitSCMSource',
            remote: "https://github.com/llvm/llvm-zorg.git",
            credentialsId: scm.userRemoteConfigs[0].credentialsId
        ])

relay.pipeline([
    "llvm.org/lnt-ctmark-aarch64-O0-g",
    "llvm.org/lnt-ctmark-aarch64-O3-flto",
    "llvm.org/lnt-ctmark-aarch64-Os",
    "llvm.org/lnt-ctmark-aarch64-Oz"
])