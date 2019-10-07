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

systemctl set-property buildslave.service TasksMax=100000

systemctl daemon-reload
service buildslave restart
