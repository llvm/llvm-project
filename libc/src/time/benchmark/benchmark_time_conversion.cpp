//===-- Benchmark for time conversion functions --------------------------===//
//
// Compares performance of update_from_seconds_fast vs unix_to_date_fast
//
//===----------------------------------------------------------------------===//

#include "../fast_date.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstring>
#include <ctime>

// Forward declare the function we want to test
namespace __llvm_libc {
namespace time_utils {
extern "C" int64_t update_from_seconds_fast(time_t total_seconds, struct tm *tm);
}
}

using namespace std::chrono;

// Test configuration
constexpr int WARMUP_ITERATIONS = 10000;
constexpr int BENCHMARK_ITERATIONS = 1000000;

// Generate diverse test timestamps covering different scenarios
std::vector<time_t> generate_test_timestamps() {
    std::vector<time_t> timestamps;
    
    // 1. Common dates (Unix epoch to Y2038)
    timestamps.push_back(0);                    // 1970-01-01 00:00:00
    timestamps.push_back(946684800);            // 2000-01-01 00:00:00
    timestamps.push_back(1000000000);           // 2001-09-09 01:46:40
    timestamps.push_back(1234567890);           // 2009-02-13 23:31:30
    timestamps.push_back(1500000000);           // 2017-07-14 02:40:00
    timestamps.push_back(1700000000);           // 2023-11-14 22:13:20
    timestamps.push_back(2000000000);           // 2033-05-18 03:33:20
    timestamps.push_back(2147483647);           // 2038-01-19 03:14:07 (32-bit max)
    
    // 2. Leap year boundaries
    timestamps.push_back(951868800);            // 2000-02-29 00:00:00 (leap year)
    timestamps.push_back(1077926400);           // 2004-02-28 00:00:00
    timestamps.push_back(1078012800);           // 2004-02-29 00:00:00 (leap year)
    timestamps.push_back(1235865600);           // 2009-02-28 00:00:00 (non-leap)
    
    // 3. Century boundaries
    timestamps.push_back(946684799);            // 1999-12-31 23:59:59
    timestamps.push_back(946684800);            // 2000-01-01 00:00:00
    
    // 4. Month boundaries
    timestamps.push_back(1609459199);           // 2020-12-31 23:59:59
    timestamps.push_back(1609459200);           // 2021-01-01 00:00:00
    
    // 5. Negative timestamps (before 1970)
    timestamps.push_back(-86400);               // 1969-12-31 00:00:00
    timestamps.push_back(-946684800);           // 1940-01-01 00:00:00
    timestamps.push_back(-2208988800);          // 1900-01-01 00:00:00
    
    // 6. Random timestamps for statistical distribution
    std::mt19937_64 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<time_t> dist(-2208988800, 2147483647);
    for (int i = 0; i < 50; i++) {
        timestamps.push_back(dist(gen));
    }
    
    return timestamps;
}

// Benchmark update_from_seconds_fast
double benchmark_update_from_seconds_fast(const std::vector<time_t>& timestamps, int iterations) {
    struct tm result;
    volatile int64_t return_code = 0;  // Prevent optimization
    
    auto start = high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (time_t ts : timestamps) {
            return_code = __llvm_libc::time_utils::update_from_seconds_fast(ts, &result);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    return static_cast<double>(duration) / (iterations * timestamps.size());
}

// Benchmark unix_to_date_fast
double benchmark_unix_to_date_fast(const std::vector<time_t>& timestamps, int iterations) {
    fast_date::DateResult result;
    
    auto start = high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; iter++) {
        for (time_t ts : timestamps) {
            result = fast_date::unix_to_date_fast(ts);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();
    
    return static_cast<double>(duration) / (iterations * timestamps.size());
}

// Verify correctness - compare outputs of both functions
bool verify_correctness(const std::vector<time_t>& timestamps) {
    int mismatches = 0;
    bool all_correct = true;
    
    for (time_t ts : timestamps) {
        struct tm tm_result;
        std::memset(&tm_result, 0, sizeof(struct tm));
        int64_t ret1 = __llvm_libc::time_utils::update_from_seconds_fast(ts, &tm_result);
        
        fast_date::DateResult fast_result = fast_date::unix_to_date_fast(ts);
        
        // Compare results
        bool match = true;
        if (ret1 == 0 && fast_result.valid) {
            // Both succeeded - compare values
            if (tm_result.tm_year != fast_result.year - 1900 ||
                tm_result.tm_mon != fast_result.month - 1 ||
                tm_result.tm_mday != fast_result.day ||
                tm_result.tm_hour != fast_result.hour ||
                tm_result.tm_min != fast_result.minute ||
                tm_result.tm_sec != fast_result.second ||
                tm_result.tm_wday != fast_result.wday ||
                tm_result.tm_yday != fast_result.yday) {
                match = false;
            }
        } else if ((ret1 != 0 && fast_result.valid) || (ret1 == 0 && !fast_result.valid)) {
            // One succeeded, other failed
            match = false;
        }
        
        if (!match) {
            mismatches++;
            all_correct = false;
            if (mismatches <= 5) {  // Only print first 5 mismatches
                std::cout << "Mismatch for timestamp " << ts << ":\n";
                std::cout << "  update_from_seconds_fast: " 
                          << (ret1 == 0 ? "success" : "error") << "\n";
                if (ret1 == 0) {
                    std::cout << "    " << (1900 + tm_result.tm_year) << "-" 
                              << std::setfill('0') << std::setw(2) << (tm_result.tm_mon + 1) << "-"
                              << std::setw(2) << tm_result.tm_mday << " "
                              << std::setw(2) << tm_result.tm_hour << ":"
                              << std::setw(2) << tm_result.tm_min << ":"
                              << std::setw(2) << tm_result.tm_sec 
                              << " (wday=" << tm_result.tm_wday << ", yday=" << tm_result.tm_yday << ")\n";
                }
                std::cout << "  unix_to_date_fast: " 
                          << (fast_result.valid ? "success" : "error") << "\n";
                if (fast_result.valid) {
                    std::cout << "    " << fast_result.year << "-" 
                              << std::setfill('0') << std::setw(2) << fast_result.month << "-"
                              << std::setw(2) << fast_result.day << " "
                              << std::setw(2) << fast_result.hour << ":"
                              << std::setw(2) << fast_result.minute << ":"
                              << std::setw(2) << fast_result.second
                              << " (wday=" << fast_result.wday << ", yday=" << fast_result.yday << ")\n";
                }
            }
        }
    }
    
    if (mismatches > 0) {
        std::cout << "\nTotal mismatches: " << mismatches << " out of " 
                  << timestamps.size() << " timestamps\n";
    }
    
    return all_correct;
}

int main() {
    std::cout << "=== Time Conversion Benchmark ===\n\n";
    
    // Generate test data
    std::vector<time_t> timestamps = generate_test_timestamps();
    std::cout << "Generated " << timestamps.size() << " test timestamps\n\n";
    
    // Verify correctness first
    std::cout << "Verifying correctness...\n";
    bool correct = verify_correctness(timestamps);
    if (correct) {
        std::cout << "✓ All results match!\n\n";
    } else {
        std::cout << "✗ Results differ - see details above\n\n";
    }
    
    // Warmup
    std::cout << "Warming up (" << WARMUP_ITERATIONS << " iterations)...\n";
    benchmark_update_from_seconds_fast(timestamps, WARMUP_ITERATIONS);
    benchmark_unix_to_date_fast(timestamps, WARMUP_ITERATIONS);
    std::cout << "Warmup complete\n\n";
    
    // Run benchmarks
    std::cout << "Running benchmarks (" << BENCHMARK_ITERATIONS << " iterations)...\n\n";
    
    double time1 = benchmark_update_from_seconds_fast(timestamps, BENCHMARK_ITERATIONS);
    std::cout << "update_from_seconds_fast: " << std::fixed << std::setprecision(2) 
              << time1 << " ns/conversion\n";
    
    double time2 = benchmark_unix_to_date_fast(timestamps, BENCHMARK_ITERATIONS);
    std::cout << "unix_to_date_fast:        " << std::fixed << std::setprecision(2) 
              << time2 << " ns/conversion\n\n";
    
    // Calculate speedup
    double speedup = time1 / time2;
    double improvement = ((time1 - time2) / time1) * 100.0;
    
    std::cout << "=== Results ===\n";
    if (speedup > 1.0) {
        std::cout << "unix_to_date_fast is " << std::fixed << std::setprecision(2) 
                  << speedup << "x FASTER (" 
                  << std::setprecision(1) << improvement << "% improvement)\n";
    } else {
        std::cout << "update_from_seconds_fast is " << std::fixed << std::setprecision(2) 
                  << (1.0 / speedup) << "x FASTER (" 
                  << std::setprecision(1) << -improvement << "% improvement)\n";
    }
    
    return correct ? 0 : 1;
}
