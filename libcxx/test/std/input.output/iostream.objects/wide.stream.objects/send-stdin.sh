# Pass a reference file as stdin to a test executable.
program=${1}
input=${2}
cat ${input} | ${program}
