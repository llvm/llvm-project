/**
 * @file dsmil_quantum_runtime.h
 * @brief Device 46 Quantum Integration Runtime (Layer 7)
 * 
 * Provides runtime support for Qiskit-based quantum simulation:
 * - QAOA/QUBO optimization for hyperparameter search
 * - Quantum feature maps for anomaly detection
 * - CPU-bound simulation (2 GB memory budget)
 * - Integration with Device 47 for hybrid workflows
 * 
 * Version: 1.0.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_QUANTUM_RUNTIME_H
#define DSMIL_QUANTUM_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_QUANTUM Device 46 Quantum Runtime
 * @{
 */

/**
 * @brief Quantum optimization problem types
 */
typedef enum {
    DSMIL_QUANTUM_QAOA,      // QAOA for combinatorial optimization
    DSMIL_QUANTUM_QUBO,      // QUBO formulation
    DSMIL_QUANTUM_VQE,       // Variational Quantum Eigensolver
    DSMIL_QUANTUM_FEATURE_MAP // Quantum feature maps for ML
} dsmil_quantum_problem_type_t;

/**
 * @brief Device 46 quantum context
 */
typedef struct {
    uint32_t device_id;           // 46
    uint8_t layer;                 // 7
    uint64_t memory_budget_bytes;  // 2 GB from Layer 7 pool
    uint32_t max_qubits;           // 8-12 qubits (statevector), ~30 (MPS)
    bool mps_enabled;              // Matrix Product State for larger circuits
    dsmil_quantum_problem_type_t problem_type;
    const char *qiskit_backend;    // "aer_simulator_statevector" or "aer_simulator_mps"
} dsmil_device46_quantum_ctx_t;

/**
 * @brief Initialize Device 46 quantum runtime
 * 
 * @param max_qubits Maximum qubits (8-12 for statevector, ~30 for MPS)
 * @param use_mps Use Matrix Product State for larger circuits
 * @return 0 on success, negative on error
 */
int dsmil_device46_quantum_init(uint32_t max_qubits, bool use_mps);

/**
 * @brief Run QAOA optimization for hyperparameter search
 * 
 * @param problem QUBO problem definition
 * @param num_vars Number of variables (≤12 for statevector)
 * @param result Output optimization result
 * @return 0 on success, negative on error
 */
int dsmil_device46_qaoa_optimize(const void *problem, uint32_t num_vars, void *result);

/**
 * @brief Generate quantum feature map for anomaly detection
 * 
 * @param data Input data
 * @param data_size Size of input data
 * @param feature_map Output quantum feature map
 * @return 0 on success, negative on error
 */
int dsmil_device46_quantum_feature_map(const void *data, size_t data_size, void *feature_map);

/**
 * @brief Hybrid workflow: Quantum-assisted model optimization
 * 
 * Integrates with Device 47 to suggest pruning/sparsity patterns
 * 
 * @param model_metadata Model metadata from Device 47
 * @param optimization_hints Output optimization suggestions
 * @return 0 on success, negative on error
 */
int dsmil_device46_hybrid_optimization(const void *model_metadata, void *optimization_hints);

/**
 * @brief Get quantum runtime context
 * 
 * @param ctx Output context
 * @return 0 on success, negative on error
 */
int dsmil_device46_get_context(dsmil_device46_quantum_ctx_t *ctx);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_QUANTUM_RUNTIME_H */
