program=${1}
expected_stdout=${2}
${program} 2>stderr.log >stdout.log
[ "${expected_stdout}" == "$(cat stdout.log)" ]
