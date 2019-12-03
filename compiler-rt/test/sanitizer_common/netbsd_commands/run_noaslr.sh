#!/bin/sh
/usr/sbin/paxctl +a "${1}"
exec "${@}"
