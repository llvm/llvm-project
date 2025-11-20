#!/bin/sh
for I in {1..3}; do \
  log show --debug --last $((SECONDS + 30))s --predicate "processID == $1" --style syslog > $2; \
  if grep -q "use-after-poison" $2; then break; fi; \
  sleep 5; \
done
