pipeline {
  agent { label 'cpp-ec2-al2-candidate' }

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  environment {
    BUILD_TYPE    = 'Release'
    LLVM_PROJECTS = 'clang;lld'   // adjust if you want more or less
    BUILD_DIR     = 'build'
  }

  stages {
    stage('Checkout') {
      steps {
        // Uses the same repo and branch that triggered the multibranch job
        checkout scm
      }
    }

    stage('Configure') {
      steps {
        sh '''
          set -eux

          mkdir -p "${BUILD_DIR}"

          cmake3 -S llvm -B "${BUILD_DIR}" -G Ninja \
            -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
            -DLLVM_ENABLE_PROJECTS="${LLVM_PROJECTS}" \
            -DCMAKE_C_COMPILER=/usr/bin/gcc10 \
            -DCMAKE_CXX_COMPILER=/usr/bin/g++10
        '''
      }
    }

    stage('Build') {
      steps {
        sh '''
          set -eux
          cmake3 --build "${BUILD_DIR}" -j"$(nproc)"
        '''
      }
    }

    // Optional, wire later once build works and you decide what to run
    // stage('Smoke tests') {
    //   steps {
    //     sh '''
    //       set -eux
    //       ctest --test-dir "${BUILD_DIR}" -R clang -j"$(nproc)" || true
    //     '''
    //   }
    // }
  }
}
