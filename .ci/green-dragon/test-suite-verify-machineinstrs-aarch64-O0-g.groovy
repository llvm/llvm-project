branchName = 'main'

library identifier: "zorg-shared-lib@${branchName}",
        retriever: modernSCM([
            $class: 'GitSCMSource',
            remote: "https://github.com/llvm/llvm-zorg.git",
            credentialsId: scm.userRemoteConfigs[0].credentialsId
        ])

common.testsuite_pipeline(label: 'macos-x86_64') {
    sh """
CMAKE_FLAGS+=" -C ../config/tasks/cmake/caches/target-arm64-iphoneos.cmake"
CMAKE_FLAGS+=" -C ../config/tasks/cmake/caches/opt-O0-g.cmake"
config/tasks/task jenkinsrun config/tasks/test-suite-verify-machineinstrs.sh -a compiler="${params.ARTIFACT}" -D CMAKE_FLAGS="\${CMAKE_FLAGS}"
    """
}
