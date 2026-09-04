#!/bin/bash

check_file() {
    local file="$1"
    if [ -z "$file" ]; then
        echo "No file passed!"
        exit 1
    fi
    if [ ! -f "$file" ]; then
        return 1
    fi

    fuser -s "$file"
    local ret=$?
    if [ $ret -eq 1 ]; then # no one has file open
        return 0
    fi
    if [ $ret -eq 0 ]; then # file open by some processes
        return 1
    fi
    if [ $ret -eq 127 ]; then
        echo "fuser command not found!"
        exit 1
    fi

    echo "Unexpected exit code $ret from fuser!"
    exit 1
}

wait_file() {
    local file="$1"
    local max_sleep=10
    check_file "$file"
    local ret=$?
    while [ $ret -ne 0 ] && [ $max_sleep -ne 0 ]; do
        sleep 1
        max_sleep=$((max_sleep - 1))
        check_file $file
        ret=$?
    done
    if [ $max_sleep -eq 0 ]; then
        echo "The file does not exist or the test hung!"
        exit 1
    fi

}
file="$1"
wait_file "$file"
