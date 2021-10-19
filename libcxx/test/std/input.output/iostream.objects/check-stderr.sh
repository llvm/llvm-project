program=${1}
expected_stderr=${2}
${program} 2>stderr.log >stdout.log
[ "${expected_stderr}" == "$(cat stderr.log)" ]
