# ex: python3 sort_yaml_functions.py
# ex: must be within yaml directory
import yaml
import os


def sort_yaml_functions(yaml_file):
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    if "functions" in yaml_data:
        yaml_data["functions"].sort(key=lambda x: x["name"])

    class IndentYamlListDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(IndentYamlListDumper, self).increase_indent(flow, False)

    with open(yaml_file, "w") as f:
        yaml.dump(
            yaml_data,
            f,
            Dumper=IndentYamlListDumper,
            default_flow_style=False,
            sort_keys=False,
        )


def main():
    current_directory = os.getcwd()
    yaml_files = [
        file for file in os.listdir(current_directory) if file.endswith(".yaml")
    ]

    for yaml_file in yaml_files:
        sort_yaml_functions(yaml_file)


if __name__ == "__main__":
    main()
