#!/bin/bash
#
# LAT5150DRVMIL Documentation Query Tool
# Simple menu-driven interface for RAG system
#

RAG_DIR="rag_system"
PYTHON="python3"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ${GREEN}LAT5150DRVMIL Documentation Query System${BLUE}          ║${NC}"
echo -e "${BLUE}║            ${YELLOW}RAG-Powered Documentation Search${BLUE}             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo

# Check if index exists
if [ ! -f "$RAG_DIR/processed_docs.json" ]; then
    echo -e "${YELLOW}⚠ Index not found. Building index from documentation...${NC}"
    echo
    $PYTHON $RAG_DIR/document_processor.py
    echo
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Failed to build index!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Index built successfully!${NC}"
    echo
fi

# Main menu
while true; do
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Main Menu:${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo
    echo "  1) 🔍 Query Documentation (TF-IDF - Legacy)"
    echo "  2) 🤖 Transformer Query (Semantic - Recommended)"
    echo "  3) 🧠 Open R1 Reasoning Query (Chain-of-Thought)"
    echo "  4) 💬 Interactive Mode (with Feedback)"
    echo "  5) 📊 Extract Structured Information"
    echo "  6) ✅ Run Test Suite (Validate Accuracy)"
    echo "  7) 📈 Show System Statistics"
    echo "  8) 🔄 Rebuild Index"
    echo "  9) 📖 Help / Documentation"
    echo "  10) 🚪 Exit"
    echo
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -n "Select option [1-10]: "
    read option

    case $option in
        1)
            echo
            echo -e "${GREEN}Enter your query (TF-IDF):${NC}"
            read -p "> " query
            echo
            echo -e "${YELLOW}Searching documentation (legacy TF-IDF)...${NC}"
            echo
            $PYTHON $RAG_DIR/rag_query.py "$query"
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        2)
            echo
            echo -e "${GREEN}Enter your query (Transformer):${NC}"
            read -p "> " query
            echo
            echo -e "${YELLOW}Searching documentation (semantic transformers)...${NC}"
            echo
            $PYTHON $RAG_DIR/transformer_query.py --no-feedback "$query"
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        3)
            echo
            echo -e "${GREEN}Enter your query (Open R1 Reasoning):${NC}"
            read -p "> " query
            echo
            echo -e "${YELLOW}Generating reasoned answer with chain-of-thought...${NC}"
            echo
            $PYTHON $RAG_DIR/openr1_query.py "$query"
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        4)
            echo
            echo -e "${GREEN}Entering interactive mode (with feedback)...${NC}"
            echo -e "${YELLOW}Type 'quit' or 'exit' to return to menu${NC}"
            echo
            $PYTHON $RAG_DIR/transformer_query.py
            clear
            ;;

        5)
            echo
            echo -e "${GREEN}Extract Structured Information${NC}"
            echo
            echo "  1) Extract specific system info"
            echo "  2) Batch extract (predefined systems)"
            echo
            read -p "Select [1-2]: " extract_option

            case $extract_option in
                1)
                    echo
                    echo -e "${GREEN}Enter system/feature name:${NC}"
                    read -p "> " system_name
                    echo
                    $PYTHON $RAG_DIR/structured_extraction.py "$system_name"
                    ;;
                2)
                    echo
                    echo -e "${YELLOW}Extracting batch systems...${NC}"
                    $PYTHON $RAG_DIR/structured_extraction.py
                    ;;
            esac
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        6)
            echo
            echo -e "${YELLOW}Running test suite...${NC}"
            echo
            echo "  1) TF-IDF baseline (legacy)"
            echo "  2) Transformer semantic search"
            echo
            read -p "Select test [1-2]: " test_option
            echo
            if [[ $test_option == "2" ]]; then
                $PYTHON $RAG_DIR/test_transformer_rag.py
            else
                $PYTHON $RAG_DIR/test_queries.py
            fi
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        7)
            echo
            echo -e "${GREEN}System Statistics:${NC}"
            echo
            $PYTHON <<EOF
from rag_system.rag_query import LAT5150RAG
import json

rag = LAT5150RAG()
stats = rag.get_stats()

print("━" * 60)
print("RAG System Statistics")
print("━" * 60)
print(f"Total Documents:    {stats['total_documents']}")
print(f"Total Chunks:       {stats['total_chunks']}")
print(f"Avg Chunks/Doc:     {stats['avg_chunks_per_doc']:.1f}")
print(f"Vocabulary Size:    {stats['vocab_size']:,} unique terms")
print(f"\nCategories: {', '.join(stats['categories'][:10])}")
if len(stats['categories']) > 10:
    print(f"... and {len(stats['categories']) - 10} more")
print("━" * 60)
EOF
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        8)
            echo
            echo -e "${YELLOW}⚠ Rebuilding index from documentation...${NC}"
            echo -e "${YELLOW}This will process all files in 00-documentation/${NC}"
            echo
            read -p "Continue? [y/N]: " confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                $PYTHON $RAG_DIR/document_processor.py
                echo
                echo -e "${GREEN}✓ Index rebuilt successfully!${NC}"
            else
                echo -e "${YELLOW}Cancelled.${NC}"
            fi
            echo
            read -p "Press Enter to continue..."
            clear
            ;;

        9)
            clear
            echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BLUE}║                    ${GREEN}Quick Help${BLUE}                           ║${NC}"
            echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
            echo
            echo -e "${GREEN}RAG System Overview:${NC}"
            echo "This tool uses Retrieval-Augmented Generation (RAG) to"
            echo "search and extract information from LAT5150DRVMIL docs."
            echo
            echo -e "${GREEN}Features:${NC}"
            echo "  • Fast TF-IDF semantic search"
            echo "  • Structured data extraction"
            echo "  • Accuracy testing and validation"
            echo "  • 235 documents, 881 searchable chunks"
            echo
            echo -e "${GREEN}Example Queries:${NC}"
            echo "  - What is DSMIL activation?"
            echo "  - How to enable NPU modules?"
            echo "  - APT41 security hardening steps?"
            echo "  - Kernel build process?"
            echo "  - ZFS upgrade procedure?"
            echo
            echo -e "${GREEN}Performance:${NC}"
            echo "  • Response time: ~2.5s average"
            echo "  • Current accuracy: ~52% (TF-IDF baseline)"
            echo "  • Target accuracy: >88% (with transformers)"
            echo
            echo -e "${GREEN}Documentation:${NC}"
            echo "  • Full guide: rag_system/README.md"
            echo "  • Source code: rag_system/*.py"
            echo
            echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
            echo
            read -p "Press Enter to return to menu..."
            clear
            ;;

        10)
            echo
            echo -e "${GREEN}Thank you for using LAT5150DRVMIL Documentation Query!${NC}"
            echo
            exit 0
            ;;

        *)
            echo
            echo -e "${RED}✗ Invalid option. Please select 1-10.${NC}"
            echo
            sleep 2
            clear
            ;;
    esac
done
