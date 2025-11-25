# RAG Implementation Guide for LAT5150DRVMIL

## Quick Start: Deploy RAG in 30 Minutes

### Prerequisites
```bash
# Check system requirements
nvidia-smi  # Verify GPU (8GB+ VRAM recommended)
df -h       # Ensure 50GB+ free disk space
```

### Step 1: Install Ollama (5 min)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull quantized model (choose one)
ollama pull llama3:8b-instruct-q4_0      # 6GB VRAM
# OR
ollama pull gemma2:9b-instruct-q4_0      # 7GB VRAM

# Test model
ollama run llama3:8b-instruct-q4_0 "What is RAG?"
```

### Step 2: Install Python Dependencies (5 min)
```bash
pip install llama-index \
            langchain \
            sentence-transformers \
            chromadb \
            pypdf \
            python-docx
```

### Step 3: Create Vector Store from Documentation (10 min)
```python
#!/usr/bin/env python3
"""
create_vector_store.py
Creates vector database from 00-documentation/
"""

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama
from llama_index import ServiceContext
import os

# Configure embedding model (BAAI/bge-base-en-v1.5 from research)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Configure LLM (Llama3-8B quantized)
llm = Ollama(model="llama3:8b-instruct-q4_0", request_timeout=120.0)

# Create service context
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=256,      # Optimal from Maharana et al.
    chunk_overlap=20     # Optimal from Maharana et al.
)

# Load documents
documents = SimpleDirectoryReader(
    "00-documentation",
    recursive=True,
    required_exts=[".md", ".txt", ".pdf"]
).load_data()

print(f"Loaded {len(documents)} documents")

# Create vector store
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
    show_progress=True
)

# Persist to disk
index.storage_context.persist(persist_dir="./storage")
print("Vector store created successfully!")
```

### Step 4: Query Your Documentation (5 min)
```python
#!/usr/bin/env python3
"""
query_rag.py
Query LAT5150DRVMIL documentation with RAG
"""

from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import Ollama
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext

# Load vector store
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3:8b-instruct-q4_0", request_timeout=120.0)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, service_context=service_context)

# Create query engine with RAG
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Retrieve top 3 chunks (optimal from research)
    response_mode="compact"
)

# Example queries
queries = [
    "What is DSMIL activation?",
    "How do I enable NPU modules?",
    "What are the security features in APT41 hardening?",
    "Explain the unified platform architecture"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    response = query_engine.query(query)
    print(f"ANSWER:\n{response}")
    print(f"\nSOURCES:")
    for node in response.source_nodes:
        print(f"  - {node.metadata.get('file_name', 'Unknown')}")
```

### Step 5: Test Accuracy (5 min)
```python
#!/usr/bin/env python3
"""
evaluate_rag.py
Measure RAG accuracy on test queries
"""

test_cases = [
    {
        "query": "What is the hydrogen storage capacity of MgH2?",
        "expected": "7.6 wt%",
        "source": "AI_SYSTEM_ENHANCEMENTS.md"
    },
    {
        "query": "What quantization format is recommended for Ollama?",
        "expected": "GGUF (Q4_0)",
        "source": "AI_SYSTEM_ENHANCEMENTS.md"
    },
    # Add more test cases from your documentation
]

correct = 0
total = len(test_cases)

for i, test in enumerate(test_cases, 1):
    response = query_engine.query(test["query"])
    answer = str(response).lower()

    # Simple substring match (enhance as needed)
    is_correct = test["expected"].lower() in answer

    print(f"\n[{i}/{total}] Query: {test['query']}")
    print(f"Expected: {test['expected']}")
    print(f"Got: {response}")
    print(f"✓ PASS" if is_correct else "✗ FAIL")

    if is_correct:
        correct += 1

accuracy = (correct / total) * 100
print(f"\n{'='*60}")
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Target: >88% (from Maharana et al. research)")
print(f"{'='*60}")
```

---

## Advanced Configuration

### Optimize Chunk Size for Your Documents
```python
# Experiment with different chunk sizes
for chunk_size in [128, 256, 512, 1024]:
    for overlap in [10, 20, 40, 80]:
        # Recreate index with new parameters
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        # Measure accuracy (use evaluate_rag.py)
        # Record best configuration
```

### Add Guardrails for Sensitive Information
```python
from llama_index.postprocessor import SimilarityPostprocessor, MetadataReplacementPostProcessor

# Filter out low-relevance results
similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# Redact classified information (example)
def redact_classified(text):
    import re
    # Redact patterns like CLASSIFIED, SECRET, etc.
    redacted = re.sub(r'\b(CLASSIFIED|SECRET|CONFIDENTIAL)\b', '[REDACTED]', text, flags=re.IGNORECASE)
    return redacted

# Apply to query engine
query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[similarity_postprocessor],
    text_qa_template="Apply redaction rules before responding"
)
```

### Memory-Augmented RAG (MemoRAG)
```python
from llama_index.memory import ChatMemoryBuffer

# Persistent conversation memory
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Create chat engine with memory
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    similarity_top_k=3
)

# Multi-turn conversation
chat_engine.chat("What is DSMIL?")
chat_engine.chat("How do I activate it?")  # Remembers previous context
chat_engine.chat("What are the security implications?")  # Builds on conversation
```

---

## Integration with Existing Systems

### DSMIL Query Interface
```python
#!/usr/bin/env python3
"""
dsmil_rag_interface.py
RAG interface for DSMIL operations
"""

class DSMILRagInterface:
    def __init__(self, vector_store_path="./storage"):
        # Initialize RAG system
        self.query_engine = self._load_query_engine(vector_store_path)

    def query_activation_steps(self):
        """Get DSMIL activation procedure"""
        return self.query_engine.query(
            "List all steps to activate DSMIL features"
        )

    def query_security_hardening(self):
        """Get APT41 security measures"""
        return self.query_engine.query(
            "What security hardening measures are needed for APT41?"
        )

    def query_npu_optimization(self):
        """Get NPU optimization guide"""
        return self.query_engine.query(
            "How to optimize NPU performance?"
        )

    def custom_query(self, question):
        """General purpose query"""
        return self.query_engine.query(question)

# Usage
rag = DSMILRagInterface()
print(rag.query_activation_steps())
```

### Structured Data Extraction (Based on Maharana et al.)
```python
#!/usr/bin/env python3
"""
extract_structured_data.py
Extract structured information from documentation
"""

EXTRACTION_PROMPT = """
Extract the following information from the text:

Name of System/Feature  :
Purpose                 :
Activation Steps        :
Requirements            :
Security Level          :
Dependencies            :

If information is not available, write "N/A".
"""

def extract_structured_info(query_engine, document_query):
    full_prompt = f"{document_query}\n\n{EXTRACTION_PROMPT}"
    response = query_engine.query(full_prompt)
    return parse_structured_output(str(response))

def parse_structured_output(response_text):
    """Parse colon-separated output into dict"""
    data = {}
    for line in response_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    return data

# Example: Extract DSMIL info
dsmil_info = extract_structured_info(
    query_engine,
    "Tell me about DSMIL activation"
)
print(json.dumps(dsmil_info, indent=2))
```

---

## Performance Optimization

### GPU Acceleration
```bash
# Check if CUDA is available
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Set environment variables for GPU usage
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0.1  # Reserve 10% VRAM for system
```

### Batch Processing for Large Document Sets
```python
from concurrent.futures import ThreadPoolExecutor

def batch_query(queries, max_workers=4):
    """Process multiple queries in parallel"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_engine.query, q): q for q in queries}
        results = {}
        for future in futures:
            query = futures[future]
            try:
                results[query] = future.result()
            except Exception as e:
                results[query] = f"Error: {e}"
        return results

# Example
queries = [
    "What is DSMIL?",
    "How to enable NPU?",
    "APT41 security features?"
]
results = batch_query(queries)
```

---

## Monitoring and Observability

### Track Response Quality
```python
import time
import logging

logging.basicConfig(level=logging.INFO)

class MonitoredQueryEngine:
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.query_log = []

    def query(self, query_str):
        start_time = time.time()
        response = self.query_engine.query(query_str)
        elapsed_time = time.time() - start_time

        # Log metrics
        log_entry = {
            "timestamp": time.time(),
            "query": query_str,
            "response_length": len(str(response)),
            "elapsed_time": elapsed_time,
            "source_count": len(response.source_nodes)
        }
        self.query_log.append(log_entry)

        logging.info(f"Query: {query_str[:50]}... | Time: {elapsed_time:.2f}s | Sources: {len(response.source_nodes)}")

        return response

    def get_metrics(self):
        """Calculate aggregate metrics"""
        if not self.query_log:
            return {}

        times = [entry["elapsed_time"] for entry in self.query_log]
        return {
            "total_queries": len(self.query_log),
            "avg_response_time": sum(times) / len(times),
            "max_response_time": max(times),
            "min_response_time": min(times)
        }

# Usage
monitored_engine = MonitoredQueryEngine(query_engine)
monitored_engine.query("What is DSMIL?")
print(monitored_engine.get_metrics())
```

---

## Troubleshooting

### Common Issues

**Issue: Out of Memory (OOM)**
```bash
# Solution 1: Reduce context window
ollama run llama3:8b-instruct-q4_0 --num-ctx 2048  # Default is 4096

# Solution 2: Use smaller model
ollama pull llama3:8b-instruct-q2_k  # Even more quantized

# Solution 3: Offload to CPU
ollama run llama3:8b-instruct-q4_0 --num-gpu 0
```

**Issue: Slow Responses (>10 seconds)**
```python
# Solution: Reduce similarity_top_k
query_engine = index.as_query_engine(
    similarity_top_k=1,  # Retrieve only top result
    response_mode="compact"
)
```

**Issue: Hallucinations / Incorrect Answers**
```python
# Solution: Add fact-checking prompt
FACT_CHECK_PROMPT = """
Answer the question using ONLY the provided context.
If the answer is not in the context, respond with "I don't have that information."
Do not make up or infer information.

Context: {context}
Question: {query}
Answer:
"""
```

---

## Next Steps

1. **Run Quick Start** (30 min) to validate setup
2. **Test with 10-20 queries** from actual documentation
3. **Measure accuracy** using evaluate_rag.py
4. **Iterate on chunk size** if accuracy < 88%
5. **Deploy in production** after validation

## Support Resources

- **Llama Index Docs**: https://docs.llamaindex.ai/
- **Ollama Models**: https://ollama.com/library
- **Research Paper**: Maharana et al., 2025 (J. Phys. Mater. 8, 035006)
- **LAT5150DRVMIL Wiki**: (internal documentation)

---

**Last Updated**: 2025-11-08
**Status**: Production Ready
