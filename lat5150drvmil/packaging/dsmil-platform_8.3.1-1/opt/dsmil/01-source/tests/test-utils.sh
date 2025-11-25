#!/bin/bash
#
# Test script for Dell MIL-SPEC userspace utilities
#

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_result="${3:-0}"
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo -n "Testing: $test_name... "
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        actual_result=0
    else
        actual_result=$?
    fi
    
    if [ $actual_result -eq $expected_result ]; then
        echo -e "${GREEN}PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}FAIL${NC} (expected $expected_result, got $actual_result)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Check if utilities exist
check_utils() {
    echo "Checking for utilities..."
    
    if [ ! -x "./milspec-control" ]; then
        echo -e "${RED}milspec-control not found or not executable${NC}"
        exit 1
    fi
    
    if [ ! -x "./milspec-monitor" ]; then
        echo -e "${RED}milspec-monitor not found or not executable${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Utilities found${NC}"
}

# Test milspec-control
test_control() {
    echo -e "\n${YELLOW}Testing milspec-control...${NC}"
    
    # Test help option
    run_test "milspec-control help" "./milspec-control -h"
    
    # Test invalid command
    run_test "milspec-control invalid command" "./milspec-control invalid" 1
    
    # Test status without driver (should fail)
    if ! lsmod | grep -q dell_milspec; then
        run_test "milspec-control status (no driver)" "./milspec-control status" 1
    else
        run_test "milspec-control status" "./milspec-control status"
    fi
    
    # Test invalid mode5 level
    run_test "milspec-control invalid mode5" "./milspec-control mode5 5" 1
    
    # Test missing argument
    run_test "milspec-control mode5 no arg" "./milspec-control mode5" 1
}

# Test milspec-monitor
test_monitor() {
    echo -e "\n${YELLOW}Testing milspec-monitor...${NC}"
    
    # Test help option
    run_test "milspec-monitor help" "./milspec-monitor -h"
    
    # Test invalid mode
    run_test "milspec-monitor invalid mode" "./milspec-monitor -m invalid" 1
    
    # Test status display
    if lsmod | grep -q dell_milspec; then
        run_test "milspec-monitor status" "./milspec-monitor -s"
    else
        run_test "milspec-monitor status (no driver)" "./milspec-monitor -s" 1
    fi
    
    # Test with timeout (run for 1 second)
    run_test "milspec-monitor timeout test" "timeout 1 ./milspec-monitor -tc || [ \$? -eq 124 ]"
}

# Memory leak test (requires valgrind)
test_memory() {
    if command -v valgrind > /dev/null 2>&1; then
        echo -e "\n${YELLOW}Testing for memory leaks...${NC}"
        
        echo -n "Testing milspec-control memory... "
        if valgrind --leak-check=full --error-exitcode=1 ./milspec-control -h > /dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
        
        echo -n "Testing milspec-monitor memory... "
        if valgrind --leak-check=full --error-exitcode=1 ./milspec-monitor -h > /dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    else
        echo -e "\n${YELLOW}Skipping memory tests (valgrind not installed)${NC}"
    fi
}

# Stress test
test_stress() {
    if lsmod | grep -q dell_milspec; then
        echo -e "\n${YELLOW}Running stress test...${NC}"
        
        echo -n "Rapid status queries (100x)... "
        local failed=0
        for i in {1..100}; do
            if ! ./milspec-control status > /dev/null 2>&1; then
                failed=$((failed + 1))
            fi
        done
        
        if [ $failed -eq 0 ]; then
            echo -e "${GREEN}PASS${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo -e "${RED}FAIL${NC} ($failed failures)"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
        TESTS_RUN=$((TESTS_RUN + 1))
    fi
}

# Summary
print_summary() {
    echo -e "\n${YELLOW}Test Summary:${NC}"
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: ${RED}$TESTS_FAILED${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "\n${RED}Some tests failed!${NC}"
        return 1
    fi
}

# Main
main() {
    echo "Dell MIL-SPEC Utilities Test Suite"
    echo "=================================="
    echo
    
    check_utils
    test_control
    test_monitor
    test_memory
    test_stress
    print_summary
}

main "$@"
