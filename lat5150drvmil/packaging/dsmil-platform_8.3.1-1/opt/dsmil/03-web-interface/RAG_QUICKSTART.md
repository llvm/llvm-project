# RAG Intelligence Database - Quick Start Guide

## What You Get

The RAG Intelligence Database panel gives you instant access to your research knowledge base directly from the military terminal interface. Add documents, search content, and see what's indexed - all with the tactical aesthetic you love.

## Where to Find It

**Location:** Right sidebar, below "Command History"
**Panel Name:** RAG INTELLIGENCE DB
**Initial State:** Collapsed (click header to expand)

## Getting Started in 3 Steps

### 1. Check What's Already Indexed

When you open the interface, RAG stats load automatically. You should see:
- APT tradecraft techniques document (5,264 tokens) already indexed
- Stats displayed in the panel header metric: `RAG DOCS: X`

**To see details:**
1. Click the "RAG INTELLIGENCE DB" header to expand panel
2. View stats: Documents, Tokens, Size
3. Scroll the "INDEXED FILES" list to see what's available

### 2. Add Your Research Papers

**Single File:**
```
1. Type or paste path: /home/john/research/apt_report.txt
2. Press Enter (or click FILE button)
3. Wait for "File added successfully" message
4. Stats auto-refresh
```

**Entire Folder:**
```
1. Type or paste path: /home/john/research/apt_papers
2. Press Enter (or click FOLDER button)
3. Wait for processing (may take time for large folders)
4. See summary: "Folder processed: 15 files, 23,412 tokens"
5. Stats auto-refresh
```

### 3. Search Your Knowledge Base

```
1. Type query: "lateral movement techniques"
2. Press Enter (or click FIND button)
3. View ranked results in chat area
4. Read content previews and source filenames
5. Reference full documents if needed
```

## Pro Tips

### Organize Your Research
```bash
# Create dedicated RAG folders
mkdir -p /home/john/research/apt_papers
mkdir -p /home/john/research/defense_tactics
mkdir -p /home/john/research/threat_intel

# Move papers into categories
mv ~/Downloads/*.pdf /home/john/research/apt_papers/

# Add entire category to RAG
# In terminal UI: /home/john/research/apt_papers
```

### Keyboard Shortcuts
- **Enter key** works in all RAG input fields (no clicking needed)
- **F4** triggers RAG query shortcut: "What documents are in the RAG system?"
- **Click header** to quickly collapse/expand panel

### File Types Supported
- `.txt` - Plain text files
- `.md` - Markdown documents
- `.pdf` - PDF documents
- Other text-based formats (check your RAG backend)

### Visual Feedback
Watch the chat area for all RAG operations:
- **Yellow pulsing**: Operation in progress
- **Green SYSTEM>**: Success messages
- **Cyan INFO>**: Detailed information
- **Red ERROR>**: Problems (with helpful descriptions)

### Stats Monitor
Keep an eye on the stats panel to track your knowledge base growth:
- **DOCUMENTS**: How many files indexed
- **TOKENS**: Total content indexed (useful for AI context)
- **SIZE**: Total data in KB

### Manual Refresh
If stats seem stale:
1. Click the **REFRESH** button in the "INDEXED FILES" section
2. Stats and document list will update immediately

## Real-World Workflow

### Daily Threat Research
```
Morning:
1. Download 3 new APT reports
2. Save to /home/john/research/daily_intel/
3. Add folder to RAG: /home/john/research/daily_intel
4. Search: "new lateral movement techniques"
5. Review results and take notes

During Work:
- Search for specific tactics as needed
- Quick reference without leaving terminal
- See which document has the answer

End of Day:
- Check stats to see knowledge base growth
- Export chat log with key findings
```

### Writing Security Report
```
1. Search: "APT29 persistence mechanisms"
2. Review results from multiple indexed papers
3. Search: "defense recommendations"
4. Cross-reference findings
5. Export chat for report references
```

### Building Tradecraft Library
```
Week 1: Add MITRE ATT&CK techniques
- Add folder: /home/john/mitre_attack/
- Verify: 134 techniques indexed

Week 2: Add APT group profiles
- Add folder: /home/john/apt_profiles/
- Verify: 23 groups indexed

Week 3: Add defense playbooks
- Add folder: /home/john/defense_docs/
- Verify: 47 playbooks indexed

Result: Comprehensive searchable knowledge base
```

## Troubleshooting

### "Error loading documents"
- Check if RAG server is running
- Verify network connection
- Try clicking REFRESH button

### "Failed to add file"
- Verify file path is correct (absolute path)
- Check file exists: `ls -la /path/to/file`
- Ensure file is readable
- Check file format is supported

### "No results found"
- Try different search terms
- Check if documents are actually indexed (view list)
- Verify documents contain relevant content
- Try broader search terms

### Stats show "ERR"
- RAG server may be offline
- Check server logs
- Try refreshing the page
- Verify backend is running

## API Endpoints (For Reference)

If you need to troubleshoot or integrate elsewhere:

```bash
# Get stats
curl http://localhost:PORT/rag/stats

# List documents
curl http://localhost:PORT/rag/list

# Add file
curl "http://localhost:PORT/rag/add-file?path=/path/to/file.txt"

# Add folder
curl "http://localhost:PORT/rag/add-folder?path=/path/to/folder"

# Search
curl "http://localhost:PORT/rag/search?q=your+query"
```

## Best Practices

### Naming Conventions
Use descriptive filenames:
- ✅ `apt29_lateral_movement_2024.txt`
- ✅ `mitre_attack_persistence_techniques.md`
- ❌ `doc1.txt`
- ❌ `paper.pdf`

### Folder Organization
Create logical hierarchies:
```
/home/john/research/
├── apt_groups/
│   ├── apt29/
│   ├── apt28/
│   └── lazarus/
├── techniques/
│   ├── lateral_movement/
│   ├── persistence/
│   └── privilege_escalation/
└── defenses/
    ├── detection/
    └── mitigation/
```

### Regular Maintenance
- **Weekly**: Add new research papers
- **Monthly**: Review document list, remove outdated content
- **Quarterly**: Reorganize folders for better categorization

### Search Tips
- Be specific: "Pass-the-Hash technique" > "hacking"
- Use quotes for exact phrases: "privilege escalation"
- Try synonyms if first search fails
- Combine multiple searches for comprehensive results

## What's Next

Once you're comfortable with RAG management:
1. Integrate with your AI queries (AI can reference indexed docs)
2. Build a comprehensive threat intelligence library
3. Create quick-reference knowledge base for incident response
4. Share folder paths with team members for standardization

## Support

If you encounter issues:
1. Check server logs for backend errors
2. Verify file permissions on indexed documents
3. Test API endpoints directly with curl
4. Review error messages in chat area
5. Check RAG_INTEGRATION_SUMMARY.md for technical details

---

**Remember:** The RAG Intelligence Database makes your research instantly searchable. The more you add, the more powerful it becomes. Start small (add a few key documents), then expand as you see the value.

**Happy researching from the tactical command center!**
