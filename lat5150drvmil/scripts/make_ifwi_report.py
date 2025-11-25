#!/usr/bin/env python3
import json, os, hashlib
from pathlib import Path

def sha256p(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None

def list_parts(root):
    parts=[]
    for p in sorted(Path(root).glob('*.bin')):
        parts.append({
            'file': str(p),
            'size': p.stat().st_size,
            'sha256': sha256p(p),
        })
    return parts

def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None

def main():
    out_dir = Path('KP14/analysis_results/dumps')
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'dump_8mb': {
            'path': '04-hardware/microcode/me_dump_8mb.bin',
            'size': Path('04-hardware/microcode/me_dump_8mb.bin').stat().st_size if Path('04-hardware/microcode/me_dump_8mb.bin').exists() else None,
            'sha256': sha256p('04-hardware/microcode/me_dump_8mb.bin'),
        },
        'dump_16mb': {
            'path': '04-hardware/microcode/me_dump_16mb.bin',
            'size': Path('04-hardware/microcode/me_dump_16mb.bin').stat().st_size if Path('04-hardware/microcode/me_dump_16mb.bin').exists() else None,
            'sha256': sha256p('04-hardware/microcode/me_dump_16mb.bin'),
        },
        'cpd_parts': list_parts('04-hardware/microcode/ifwi_parts'),
        'bg_windows': list_parts('04-hardware/microcode/ifwi_parts/bg'),
        'kp14_parts': {},
    }
    # Attach KP14 JSONs if present
    kp_parts_dir = Path('KP14/analysis_results/ifwi_parts')
    kp_dump_dir = Path('KP14/analysis_results/dumps')
    for p in sorted(kp_parts_dir.glob('*.json')):
        report['kp14_parts'][p.name] = load_json(p)
    rep_path = out_dir / 'ifwi_summary.json'
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote IFWI summary: {rep_path}")

if __name__ == '__main__':
    main()

