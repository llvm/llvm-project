branchName = 'main'

library identifier: "zorg-shared-lib@${branchName}",
        retriever: modernSCM([
            $class: 'GitSCMSource',
            remote: "https://github.com/llvm/llvm-zorg.git",
            credentialsId: scm.userRemoteConfigs[0].credentialsId
        ])

common.testsuite_pipeline(label: 'macos-x86_64') {
    timeout(30) {
        sh """
LNT_FLAGS+=" -C config/tasks/cmake/caches/target-arm64-iphoneos.cmake"
LNT_FLAGS+=" -C config/tasks/cmake/caches/opt-O3-flto.cmake"
config/tasks/task jenkinsrun config/tasks/lnt-ctmark.sh -a compiler="${params.ARTIFACT}" -D LNT_FLAGS="\${LNT_FLAGS}"
        """
    }
}
