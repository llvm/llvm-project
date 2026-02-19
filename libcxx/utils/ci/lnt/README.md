This directory contains utilities for continuous benchmarking of libc++ with LNT.
This can be done locally using a local instance, or using a public instance like http://lnt.llvm.org.

Example for running locally:

```
# Create an instance and run a server
lnt create my-instance
echo "api_auth_token = 'example_token'" >> my-instance/lnt.cfg
lnt runserver my-instance

# In another terminal, create the libcxx test suite on the locally-running server
cat <<EOF > lnt-admin-config.yaml
lnt_url: "http://localhost:8000"
database: default
auth_token: example_token
EOF
lnt admin --config lnt-admin-config.yaml --testsuite libcxx test-suite add libcxx/utils/ci/lnt/schema.yaml

# Then, watch for libc++ commits and submit benchmark results to the locally-running instance
libcxx/utils/ci/lnt/commit-watch --lnt-url http://localhost:8000 --test-suite libcxx --machine my-laptop |      \
    while read commit; do                                                                                       \
        libcxx/utils/ci/lnt/run-benchmarks                                                                      \
            --test-suite-commit abcdef09                                                                        \
            --lnt-url http://localhost:8000                                                                     \
            --machine my-laptop                                                                                 \
            --test-suite libcxx                                                                                 \
            --compiler clang++                                                                                  \
            --benchmark-commit ${commit}                                                                        \
    done
```
