pipeline {
  agent { label 'cpp-ec2-al2' }

  options {
    timestamps()
    disableConcurrentBuilds()
  }

  environment {
    BUILD_TYPE    = 'Release'
    LLVM_PROJECTS = 'clang;lld'
    BUILD_DIR     = 'build'

    // Use GCC 10 toolchain installed on the AMI
    CC  = 'gcc10-gcc'
    CXX = 'gcc10-g++'
  }

  stages {

    stage('Toolchain overview') {
      steps {
        sh '''
          set -eux

          echo "=== compilers ==="
          which gcc || true
          gcc --version || true
          which g++ || true
          g++ --version || true

          echo "=== GCC 10 toolset ==="
          which gcc10-gcc || true
          gcc10-gcc --version || true
          which gcc10-g++ || true
          gcc10-g++ --version || true

          echo "=== clang ==="
          which clang || true
          clang --version || true

          echo "=== python ==="
          which python || true
          python --version || true
          which python3 || true
          python3 --version || true
          which python3.8 || true
          python3.8 --version || true

          echo "=== cmake ==="
          which cmake || true
          cmake --version || true
          which cmake3 || true
          cmake3 --version || true

          echo "=== ninja ==="
          which ninja || true
          ninja --version || true
          which ninja-build || true
          ninja-build --version || true

          echo "=== pkg-config ==="
          which pkg-config || true
          pkg-config --version || true
        '''
      }
    }

    stage('Checkout') {
      steps {
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
            -DCMAKE_C_COMPILER="${CC}" \
            -DCMAKE_CXX_COMPILER="${CXX}" \
            -DPython3_EXECUTABLE=/usr/bin/python3.8
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
