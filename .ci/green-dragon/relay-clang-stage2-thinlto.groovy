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
    "llvm.org/clang-stage2-Rthinlto"
])