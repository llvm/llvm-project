"""Print a NullabilitySafety TU summary as sorted lines, independent of
entity ids:

    <contributor> <set> <entity>

Entities are USRs; a parameter is followed by " param <n>" and a function
return by " return". ConditionalEvidence prints one line per source as
"<assignee> <- <source>".
"""
import json
import sys


def name(entry):
    usr, suffix = entry["usr"], entry["suffix"]
    if suffix == "0":
        return usr + " return"
    if suffix:
        return usr + " param " + suffix
    return usr


def main():
    summary = json.load(open(sys.argv[1]))
    ids = {e["id"]: name(e["name"]) for e in summary["id_table"]}
    lines = []
    for data in summary["data"]:
        if data["summary_name"] != "NullabilitySafety":
            continue
        for s in data["summary_data"]:
            contributor = ids[s["entity_id"]]
            def epl(e):
                entity, level = e
                suffix = "" if level == 1 else " level %d" % level
                return ids[entity["@"]] + suffix

            for key, epls in s["entity_summary"].items():
                if key == "ConditionalEvidence":
                    for assignee, *sources in epls:
                        for source in sources:
                            lines.append("%s %s %s <- %s" % (
                                contributor, key, epl(assignee), epl(source)))
                    continue
                for e in epls:
                    lines.append("%s %s %s" % (contributor, key, epl(e)))
    print("\n".join(sorted(lines)))


main()
