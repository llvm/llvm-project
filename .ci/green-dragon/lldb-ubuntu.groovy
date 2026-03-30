pipeline {
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30'))
        skipDefaultCheckout()
    }

    parameters {
        string(name: 'BUILD_TYPE', defaultValue: params.BUILD_TYPE ?: 'Release', description: 'Default CMake build type; one of: Release, Debug, ...')
    }

    agent {
        node {
            label env.JOB_NAME.contains('aarch64') ? 'linux-aarch64' : 'linux-x86_64'
        }
    }

    stages {
        stage('Setup Docker') {
            steps {
                withCredentials([string(credentialsId: 'aws_account', variable: 'AWS_ACCOUNT_ID')]) {
                    script {
                        env.AWS_DEFAULT_REGION = 'us-west-2'
                        env.DOCKER_SERVER = "${AWS_ACCOUNT_ID}.dkr.ecr.${env.AWS_DEFAULT_REGION}.amazonaws.com"
                        def tag = env.JOB_NAME.contains('aarch64')
                            ? 'main-ci-ecr-8ce5c7b:swift-ci-ubuntu2404-aarch64'
                            : 'main-ci-ecr-8ce5c7b:swift-ci-ubuntu2404'
                        env.DOCKER_IMAGE = "${env.DOCKER_SERVER}/${tag}"
                    }
                    sh "aws ecr get-login-password --region ${env.AWS_DEFAULT_REGION} | docker login --username AWS --password-stdin ${env.DOCKER_SERVER} 2>/dev/null"
                    sh "docker pull ${env.DOCKER_IMAGE}"
                }
            }
        }

        stage('Checkout') {
            steps {
                timeout(30) {
                    dir('llvm-project') {
                        checkout scm
                    }
                }
            }
        }

        stage('Print Machine Info') {
            steps {
                sh """
                    docker run --rm \\
                        ${env.DOCKER_IMAGE} \\
                        bash -c "cmake --version; ninja --version; python3 --version; swig -version"
                """
            }
        }

        stage('Build and Test') {
            steps {
                timeout(240) {
                    catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                        writeFile file: 'build.sh', text: '''#!/usr/bin/env bash
set -ex

/usr/bin/cmake -G Ninja \
    -S /workspace/llvm-project/llvm \
    -B /workspace/llvm-build \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DLLDB_ENABLE_CURSES=ON \
    -DLLDB_ENABLE_LIBXML2=ON \
    -DLLDB_ENABLE_LUA=OFF \
    -DLLDB_ENABLE_LZMA=OFF \
    -DLLDB_ENABLE_PYTHON=ON \
    -DLLDB_ENABLE_SWIG=ON \
    -DLLVM_BUILD_TOOLS=TRUE \
    -DLLVM_ENABLE_ASSERTIONS:BOOL=TRUE \
    -DLLVM_ENABLE_LIBEDIT=FORCE_ON \
    -DLLVM_ENABLE_LIBXML2=FORCE_ON \
    -DLLVM_BUILD_UTILS=TRUE \
    -DLLVM_ENABLE_PROJECTS="clang;lldb;lld" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind;compiler-rt" \
    -DLLVM_OPTIMIZED_TABLEGEN:BOOL=TRUE \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DLLVM_USE_SPLIT_DWARF=TRUE \
    "-DLLVM_LIT_ARGS=-v --time-tests --param color_output --xunit-xml-output=/workspace/llvm-build/test/results.xml" \
    -DLLDB_ENFORCE_STRICT_TEST_REQUIREMENTS=ON

ninja -C /workspace/llvm-build check-lldb
'''
                        sh """
                            mkdir -p llvm-build/test
                            chmod -R 777 llvm-build
                            chmod +x build.sh
                            docker run --rm \\
                                --security-opt=no-new-privileges \\
                                --cap-add=SYS_PTRACE \\
                                --security-opt seccomp=unconfined \\
                                -e BUILD_TYPE=${params.BUILD_TYPE} \\
                                -v "\${WORKSPACE}:/workspace" \\
                                ${env.DOCKER_IMAGE} \\
                                bash /workspace/build.sh
                        """
                    }
                }
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'llvm-build/test/results.xml'
        }
        cleanup {
            deleteDir()
        }
    }
}
