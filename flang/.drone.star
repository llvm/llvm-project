def clang(arch):
    return {
        "kind": "pipeline",
        "name": "%s-clang" % arch,
        "steps": [
            {
                "name": "test",
                "image": "ubuntu",
                "commands": [
                    "apt-get update && apt-get install -y clang-8 cmake ninja-build lld-8 llvm-8-dev libc++-8-dev libc++abi-8-dev libz-dev",
                    "mkdir build && cd build",
                    'env CC=clang-8 CXX=clang++-8 CXXFLAGS="-UNDEBUG" LDFLAGS="-fuse-ld=lld" cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..',
                    "ninja -j8",
                    "ctest --output-on-failure -j24",
                ],
            },
        ],

    }

def gcc(arch):
    return {
            "kind": "pipeline",
            "name": "%s-gcc" % arch,
            "steps": [
                {
                    "name": "test",
                    "image": "gcc",
                    "commands": [
			"apt-get update && apt-get install -y cmake ninja-build llvm-dev libz-dev",
                        "mkdir build && cd build",
                        'env CC=gcc CXX=g++ CXXFLAGS="-UNDEBUG" LDFLAGS="-fuse-ld=gold" cmake -GNinja -DCMAKE_BUILD_TYPE=Release ..',
                        "ninja -j8",
                        "ctest --output-on-failure -j24",
                    ],
                },
            ],

        }

def main(ctx):
    return [
        clang("amd64"),
        clang("arm64"),
        gcc("amd64"),
        gcc("arm64"),
    ]

