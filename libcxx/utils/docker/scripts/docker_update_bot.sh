#!/usr/bin/env bash

set -x

cd /libcxx
git pull


#pushd /tmp
#curl -sSO https://dl.google.com/cloudagents/install-monitoring-agent.sh
#bash install-monitoring-agent.sh
#curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
#bash install-logging-agent.sh --structured
#popd


apt-get update -y
apt-get upgrade -y

apt-get install sudo -y

# FIXME(EricWF): Remove this hack. It's only in place to temporarily fix linking libclang_rt from the
# debian packages.
# WARNING: If you're not a buildbot, DO NOT RUN!
apt-get install lld-9
rm /usr/bin/ld
ln -s /usr/bin/lld-9 /usr/bin/ld

systemctl set-property buildslave.service TasksMax=100000

systemctl daemon-reload
service buildslave restart
