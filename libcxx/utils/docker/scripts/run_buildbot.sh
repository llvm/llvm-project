#!/usr/bin/env bash
set -x

readonly BOT_ROOT=/b
readonly BOT_ROOT_NAME=$1
readonly BOT_PASS=$2

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

function setup_numbered_bot() {
  local BOT_NAME=$1
  local BOT_DIR=$2
  mkdir -p $BOT_DIR

  buildslave stop $BOT_DIR
  chown buildbot:buildbot $BOT_DIR
  rm -rf $BOT_DIR/*

  buildslave create-slave --allow-shutdown=signal "$BOT_DIR" "lab.llvm.org:9990" "$BOT_NAME" "$BOT_PASS"

  echo "Eric Fiselier <ericwf@google.com>" > $BOT_DIR/info/admin

  echo "Connecting as $1"
  {
    uname -a | head -n1
    cmake --version | head -n1
    g++ --version | head -n1
    ld --version | head -n1
    date
    lscpu
  } > $BOT_DIR/info/host


#echo "SLAVE_RUNNER=/usr/bin/buildslave
#SLAVE_ENABLED[1]=\"1\"
#SLAVE_NAME[1]=\"$BOT_NAME\"
#SLAVE_USER[1]=\"buildbot\"
#SLAVE_BASEDIR[1]=\"$BOT_DIR\"
#SLAVE_OPTIONS[1]=\"\"
#SLAVE_PREFIXCMD[1]=\"\"" > $BOT_DIR/buildslave.cfg

  ls $BOT_DIR/
  cat $BOT_DIR/buildbot.tac
}

function try_start_builder {
  local N=$1
  local BOT_DIR="$BOT_ROOT/b$N"
  local BOT_NAME="$BOT_ROOT_NAME$N"

  systemctl daemon-reload
  service buildslave restart
  setup_numbered_bot "$BOT_NAME" "$BOT_DIR"

  systemctl daemon-reload
  service buildslave restart

  chown -R buildbot:buildbot $BOT_DIR/
  sudo -u buildbot /usr/bin/buildslave start $BOT_DIR/

  sleep 30
  cat $BOT_DIR/twistd.log
  if grep --quiet "slave is ready" $BOT_DIR/twistd.log; then
    return 0
  fi
  if grep --quiet "configuration update complete" $BOT_DIR/twistd.log; then
    return 0
  fi
  if grep "rejecting duplicate slave" $BOT_DIR/twistd.log; then
    return 1
  fi
  echo "Unknown error"
  cat $BOT_DIR/twistd.log
  exit 1
}

for N in `shuf -i 1-5`
do
  if try_start_builder $N; then
    break
  fi
  echo "failed to start any buildbot"
  shutdown now
done

# GCE can restart instance after 24h in the middle of the build.
# Gracefully restart before that happen.
sleep 72000
while pkill -SIGHUP buildslave; do sleep 5; done;
shutdown now
