# Check that the stderr of the executed program matches a reference file.
program=${1}
expected_file=${2}
${program} 2>stderr.log >stdout.log
cmp stderr.log "${expected_file}"
