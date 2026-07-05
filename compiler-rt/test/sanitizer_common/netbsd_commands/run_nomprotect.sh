#!/bin/sh
/usr/sbin/paxctl +m "${1}"
exec "${@}"
