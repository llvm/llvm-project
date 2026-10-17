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

jobs = [
    "llvm.org/test-suite-verify-machineinstrs-x86_64-O0-g",
    "llvm.org/test-suite-verify-machineinstrs-x86_64-O3",
    "llvm.org/test-suite-verify-machineinstrs-x86_64h-O3",
    "llvm.org/test-suite-verify-machineinstrs-aarch64-globalisel-O0-g",
    "llvm.org/test-suite-verify-machineinstrs-aarch64-O0-g",
    "llvm.org/test-suite-verify-machineinstrs-aarch64-O3"
]

relay.pipeline jobs
