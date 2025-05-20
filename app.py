from flask import Flask, render_template, request
import subprocess
import yaml
import threading
import os

app = Flask(__name__)

CONFIG_PATH = "config.yaml"
CHECK_SCRIPTS = [
    "remote_class_check.py",
    "remote_header_check.py",
    "remote_naming_conventions.py",
    "remote_clang_format.py",
    "remote_clang_tidy.py"
]

results = {}

def run_script(script):
    try:
        output = subprocess.check_output(["python3", script], stderr=subprocess.STDOUT)
        results[script] = {"status": "✅ Passed", "output": output.decode("utf-8")}
    except subprocess.CalledProcessError as e:
        results[script] = {"status": "❌ Failed", "output": e.output.decode("utf-8")}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pr_number = request.form['pr_number']

        # Update config.yaml
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        config['project']['pr_number'] = int(pr_number)
        with open(CONFIG_PATH, 'w') as f:
            yaml.safe_dump(config, f)

        # Clear previous results
        results.clear()

        # Run all scripts
        threads = []
        for script in CHECK_SCRIPTS:
            t = threading.Thread(target=run_script, args=(script,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        return render_template("results.html", results=results, pr_number=pr_number)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
