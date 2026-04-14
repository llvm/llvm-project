pipeline {
    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30'))
        skipDefaultCheckout()
    }

    parameters {
        string(name: 'LABEL', defaultValue: params.LABEL ?: 'windows-server-2019', description: 'Node label to run on')

        string(name: 'BUILD_TYPE', defaultValue: params.BUILD_TYPE ?: 'Release', description: 'Default CMake build type; one of: Release, Debug, ...')
    }

    agent {
        node {
            label params.LABEL
        }
    }

    stages {
        stage('Pull Docker Image') {
            steps {
                bat 'docker pull swiftlang/swift-ci:lldb-windowsservercore-1809'
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
                bat '''
                    docker run --rm ^
                        swiftlang/swift-ci:lldb-windowsservercore-1809 ^
                        powershell -Command "cmake --version; ninja --version; python --version; swig -version"
                '''
            }
        }

        stage('Build and Test') {
            steps {
                timeout(240) {
                    catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                        writeFile file: 'build.bat', text: '''@echo off
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" || exit /b 1

cmake -G Ninja ^
    -S llvm ^
    -B ..\\llvm-build\\ ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_INSTALL_PREFIX=..\\llvm-install\\base ^
    -DCMAKE_TOOLCHAIN_FILE=C:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake ^
    -DLLVM_ENABLE_PROJECTS="clang;lldb" ^
    -DLLVM_ENABLE_ASSERTIONS=ON ^
    -DLLVM_ENABLE_LIBEDIT=OFF ^
    -DLLVM_OPTIMIZED_TABLEGEN=ON ^
    -DLLVM_BUILD_TOOLS=ON ^
    -DLLVM_BUILD_UTILS=ON ^
    -DLLVM_ENABLE_LIBXML2=FORCE_ON ^
    -DLLDB_ENABLE_SWIG=ON ^
    -DLLDB_ENABLE_PYTHON=ON ^
    -DLLDB_ENABLE_LUA=OFF ^
    -DLLDB_ENABLE_LIBXML2=ON ^
    -DLLVM_TARGETS_TO_BUILD=Native ^
    -DLLVM_LIT_ARGS="-v --time-tests --xunit-xml-output=C:\\workspace\\llvm-build\\test\\results.xml" ^
    -DPython3_EXECUTABLE="C:\\Program Files\\Python313\\python.exe" || exit /b 1

if not exist ..\\llvm-build\\test mkdir ..\\llvm-build\\test

ninja check-lldb -C ..\\llvm-build || exit /b 1
'''
                        bat '''
                            docker run --rm ^
                                -e BUILD_TYPE=%BUILD_TYPE% ^
                                -v "%CD%:C:\\workspace" ^
                                -w "C:\\workspace\\llvm-project" ^
                                swiftlang/swift-ci:lldb-windowsservercore-1809 ^
                                cmd /C C:\\workspace\\build.bat
                        '''
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
