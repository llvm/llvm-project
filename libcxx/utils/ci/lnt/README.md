This directory contains utilities for continuous benchmarking of libc++ with LNT.
This can be done locally using a local instance, or using a public instance like http://lnt.llvm.org.

## Running a benchmark bot

The `run-benchbot` script is the main entry point for running benchmarks. That script
is where libc++'s pre-defined LNT bot configurations are defined. To benchmark specific
commits:

```
libcxx/utils/ci/lnt/run-benchbot --llvm-root <monorepo> <builder> -- <commit1> <commit2> ...
```

Results are stored as JSON files in `<llvm-root>/build/<builder>/` by default. Use
`--build-dir <dir>` to override the output directory.

To continuously poll for un-benchmarked commits and submit results to a LNT instance:

```
libcxx/utils/ci/lnt/run-benchbot --llvm-root <monorepo> --lnt-url http://lnt.llvm.org <builder>
```

## Setting up a local LNT instance

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

# Then run the benchbot against the local instance
libcxx/utils/ci/lnt/run-benchbot --llvm-root <monorepo> --lnt-url http://localhost:8000 <builder>
```
