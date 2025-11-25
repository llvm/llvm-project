def _format_web_results(self, search_results):
    """Format web search results for AI context"""
    context = f"Web search results for: {search_results['query']}\n\n"

    for i, result in enumerate(search_results['results'], 1):
        context += f"[{i}] {result['title']}\n"
        context += f"    {result['snippet']}\n"
        context += f"    URL: {result['url']}\n\n"

    return context
