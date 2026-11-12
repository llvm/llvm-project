#!/bin/bash

# This serves as the entrypoint for docker to allow us to
# run and start the buildbot while supplying the password
# as an argument.
buildbot-worker create-worker --keepalive=200 "${WORKER_NAME}" \
  lab.llvm.org:9990 "${WORKER_NAME}" "$1"

buildbot-worker start "${WORKER_NAME}"
tail -f ${WORKER_NAME}/twistd.log
