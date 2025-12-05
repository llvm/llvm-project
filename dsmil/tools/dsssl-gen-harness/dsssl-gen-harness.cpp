/**
 * @file dsssl-gen-harness.cpp
 * @brief DSSSL Fuzz Harness Generator
 *
 * Generates libFuzzer/AFL++ harnesses for TLS, X.509, and state machine fuzzing.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>
#include <cstdlib>

struct HarnessConfig {
    std::string type;
    std::string role;
    bool use_0rtt = false;
    bool use_tickets = false;
    bool use_psk = false;
    size_t max_record_size = 16384;
    size_t max_chain_len = 8;
    bool fuzz_name_constraints = false;
    bool fuzz_idn = false;
    bool fuzz_tickets = false;
    bool fuzz_psk_binding = false;
    bool fuzz_0rtt = false;
};

void generate_tls_dialect_harness(const HarnessConfig &config, std::ostream &out) {
    out << "// Generated TLS Dialect Harness\n";
    out << "#include <stdint.h>\n";
    out << "#include <stddef.h>\n";
    out << "#include \"dsssl_fuzz_telemetry.h\"\n";
    out << "\n";
    out << "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {\n";
    out << "    // Initialize telemetry\n";
    out << "    static bool initialized = false;\n";
    out << "    if (!initialized) {\n";
    out << "        dsssl_fuzz_telemetry_init(NULL, 65536);\n";
    out << "        initialized = true;\n";
    out << "    }\n";
    out << "\n";
    out << "    // Set context ID (hash of input)\n";
    out << "    uint64_t context_id = 0;\n";
    out << "    for (size_t i = 0; i < size && i < 8; i++) {\n";
    out << "        context_id = (context_id << 8) | data[i];\n";
    out << "    }\n";
    out << "    dsssl_fuzz_set_context(context_id);\n";
    out << "\n";
    out << "    // Parse fuzzer input as TLS handshake parameters\n";
    out << "    if (size < 1) return 0;\n";
    out << "\n";
    out << "    // Extract parameters from input\n";
    out << "    uint8_t version_major = size > 0 ? data[0] : 3;\n";
    out << "    uint8_t version_minor = size > 1 ? data[1] : 3;\n";
    out << "    uint8_t cipher_suite_count = size > 2 ? (data[2] % 32) : 1;\n";
    out << "\n";
    out << "    // Build synthetic TLS handshake\n";
    out << "    // TODO: Implement actual TLS handshake construction\n";
    out << "    // This would use DSSSL/OpenSSL APIs to build handshake\n";
    out << "\n";
    if (config.use_0rtt) {
        out << "    // Handle 0-RTT if enabled\n";
        out << "    if (size > 10 && (data[3] & 0x01)) {\n";
        out << "        // Process 0-RTT data\n";
        out << "    }\n";
    }
    if (config.use_tickets) {
        out << "    // Handle session tickets if enabled\n";
        out << "    if (size > 10 && (data[3] & 0x02)) {\n";
        out << "        dsssl_ticket_event(DSSSL_TICKET_ISSUE, context_id);\n";
        out << "    }\n";
    }
    out << "\n";
    out << "    // Flush telemetry\n";
    out << "    dsssl_fuzz_clear_events();\n";
    out << "\n";
    out << "    return 0;\n";
    out << "}\n";
}

void generate_x509_pki_harness(const HarnessConfig &config, std::ostream &out) {
    out << "// Generated X.509 PKI Harness\n";
    out << "#include <stdint.h>\n";
    out << "#include <stddef.h>\n";
    out << "#include \"dsssl_fuzz_telemetry.h\"\n";
    out << "\n";
    out << "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {\n";
    out << "    // Initialize telemetry\n";
    out << "    static bool initialized = false;\n";
    out << "    if (!initialized) {\n";
    out << "        dsssl_fuzz_telemetry_init(NULL, 65536);\n";
    out << "        initialized = true;\n";
    out << "    }\n";
    out << "\n";
    out << "    // Set context ID\n";
    out << "    uint64_t context_id = 0;\n";
    out << "    for (size_t i = 0; i < size && i < 8; i++) {\n";
    out << "        context_id = (context_id << 8) | data[i];\n";
    out << "    }\n";
    out << "    dsssl_fuzz_set_context(context_id);\n";
    out << "\n";
    out << "    // Parse ASN.1 DER structures from input\n";
    out << "    if (size < 10) return 0;\n";
    out << "\n";
    out << "    // Extract certificate chain length\n";
    out << "    size_t chain_len = (data[0] % " << config.max_chain_len << ") + 1;\n";
    out << "\n";
    out << "    // Build certificate chain\n";
    out << "    // TODO: Implement actual X.509 parsing and path building\n";
    out << "    // This would use DSSSL/OpenSSL X.509 APIs\n";
    out << "\n";
    if (config.fuzz_name_constraints) {
        out << "    // Fuzz name constraints\n";
        out << "    // TODO: Extract and apply name constraints\n";
    }
    if (config.fuzz_idn) {
        out << "    // Fuzz IDN (Internationalized Domain Names)\n";
        out << "    // TODO: Extract and validate IDN\n";
    }
    out << "\n";
    out << "    // Record PKI decision\n";
    out << "    dsssl_telemetry_event_t ev = {0};\n";
    out << "    ev.event_type = DSSSL_EVENT_PKI_DECISION;\n";
    out << "    ev.data.pki_decision.decision = \"accept\";  // Would be actual result\n";
    out << "    ev.data.pki_decision.chain_len = chain_len;\n";
    out << "\n";
    out << "    return 0;\n";
    out << "}\n";
}

void generate_tls_state_harness(const HarnessConfig &config, std::ostream &out) {
    out << "// Generated TLS State Machine Harness\n";
    out << "#include <stdint.h>\n";
    out << "#include <stddef.h>\n";
    out << "#include \"dsssl_fuzz_telemetry.h\"\n";
    out << "\n";
    out << "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {\n";
    out << "    // Initialize telemetry\n";
    out << "    static bool initialized = false;\n";
    out << "    if (!initialized) {\n";
    out << "        dsssl_fuzz_telemetry_init(NULL, 65536);\n";
    out << "        initialized = true;\n";
    out << "    }\n";
    out << "\n";
    out << "    // Set context ID\n";
    out << "    uint64_t context_id = 0;\n";
    out << "    for (size_t i = 0; i < size && i < 8; i++) {\n";
    out << "        context_id = (context_id << 8) | data[i];\n";
    out << "    }\n";
    out << "    dsssl_fuzz_set_context(context_id);\n";
    out << "\n";
    out << "    if (size < 1) return 0;\n";
    out << "\n";
    out << "    // Simulate TLS state machine sequences\n";
    out << "    uint8_t sequence_type = data[0] % 4;\n";
    out << "\n";
    out << "    switch (sequence_type) {\n";
    out << "    case 0:  // Initial handshake\n";
    out << "        dsssl_state_transition(1, 0, 1);  // TLS handshake SM\n";
    out << "        break;\n";
    if (config.fuzz_tickets) {
        out << "    case 1:  // Ticket issuance\n";
        out << "        dsssl_ticket_event(DSSSL_TICKET_ISSUE, context_id);\n";
        out << "        break;\n";
        out << "    case 2:  // Ticket use\n";
        out << "        dsssl_ticket_event(DSSSL_TICKET_USE, context_id);\n";
        out << "        break;\n";
    }
    if (config.fuzz_0rtt) {
        out << "    case 3:  // 0-RTT attempt\n";
        out << "        dsssl_state_transition(2, 0, 1);  // 0-RTT SM\n";
        out << "        break;\n";
    }
    out << "    }\n";
    out << "\n";
    out << "    return 0;\n";
    out << "}\n";
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml> <output.cpp>\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string output_path = argv[2];

    try {
        YAML::Node config = YAML::LoadFile(config_path);
        
        HarnessConfig harness_config;
        
        if (config["targets"]) {
            auto targets = config["targets"];
            auto first_target = targets.begin();
            
            if (first_target != targets.end()) {
                auto target = first_target->second;
                harness_config.type = target["type"].as<std::string>("tls_handshake");
                harness_config.role = target["role"].as<std::string>("client");
                
                if (target["use_0rtt"]) {
                    harness_config.use_0rtt = target["use_0rtt"].as<bool>();
                }
                if (target["use_tickets"]) {
                    harness_config.use_tickets = target["use_tickets"].as<bool>();
                }
                if (target["max_record_size"]) {
                    harness_config.max_record_size = target["max_record_size"].as<size_t>();
                }
                if (target["max_chain_len"]) {
                    harness_config.max_chain_len = target["max_chain_len"].as<size_t>();
                }
                if (target["fuzz_name_constraints"]) {
                    harness_config.fuzz_name_constraints = target["fuzz_name_constraints"].as<bool>();
                }
                if (target["fuzz_idn"]) {
                    harness_config.fuzz_idn = target["fuzz_idn"].as<bool>();
                }
                if (target["fuzz_tickets"]) {
                    harness_config.fuzz_tickets = target["fuzz_tickets"].as<bool>();
                }
                if (target["fuzz_psk_binding"]) {
                    harness_config.fuzz_psk_binding = target["fuzz_psk_binding"].as<bool>();
                }
                if (target["fuzz_0rtt"]) {
                    harness_config.fuzz_0rtt = target["fuzz_0rtt"].as<bool>();
                }
            }
        }

        std::ofstream out(output_path);
        if (!out.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_path << "\n";
            return 1;
        }

        if (harness_config.type == "tls_handshake") {
            generate_tls_dialect_harness(harness_config, out);
        } else if (harness_config.type == "x509_path") {
            generate_x509_pki_harness(harness_config, out);
        } else if (harness_config.type == "tls_state_machine") {
            generate_tls_state_harness(harness_config, out);
        } else {
            std::cerr << "Error: Unknown harness type: " << harness_config.type << "\n";
            return 1;
        }

        std::cout << "Generated harness: " << output_path << "\n";
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
