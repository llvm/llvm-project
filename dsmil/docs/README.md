# DSLLVM Documentation Index

**Version**: 1.6.0 (High-Assurance Phase)
**Last Updated**: November 2024

Welcome to the DSLLVM comprehensive documentation. This directory contains all design specifications, feature guides, integration instructions, and reference materials for the Defense Semantic Language & LLVM (DSLLVM) war-fighting compiler.

---

## 📚 Documentation Organization

### Core Architecture & Design

**Foundation documents** - Start here to understand DSLLVM's architecture and vision

| Document | Description | Audience |
|----------|-------------|----------|
| [DSLLVM-DESIGN.md](DSLLVM-DESIGN.md) | Complete design specification and architecture | Engineers, Architects |
| [DSLLVM-ROADMAP.md](DSLLVM-ROADMAP.md) | Strategic roadmap (v1.0 → v2.0) | Project Managers, Leadership |
| [ATTRIBUTES.md](ATTRIBUTES.md) | Complete attribute reference guide | Developers |
| [PIPELINES.md](PIPELINES.md) | Pass pipeline configurations | Compiler Engineers |
| [PATH-CONFIGURATION.md](PATH-CONFIGURATION.md) | Dynamic path configuration and portable installations | DevOps, System Administrators |

---

## 🎯 Feature Guides (By Version)

### v1.3: Operational Control (Complete ✅)

**Mission Profile System**

| Document | Feature | Description |
|----------|---------|-------------|
| [MISSION-PROFILES-GUIDE.md](MISSION-PROFILES-GUIDE.md) | Feature 1.1 | Mission profiles for border ops, cyber defense, exercises |
| [MISSION-PROFILE-PROVENANCE.md](MISSION-PROFILE-PROVENANCE.md) | Feature 1.1 | Provenance integration with mission profiles |

**Fuzzing & Testing**

| Document | Feature | Description |
|----------|---------|-------------|
| [FUZZ-HARNESS-SCHEMA.md](FUZZ-HARNESS-SCHEMA.md) | Feature 1.2 | Auto-generated fuzz harness schema |
| [FUZZ-CICD-INTEGRATION.md](FUZZ-CICD-INTEGRATION.md) | Feature 1.2 | CI/CD fuzzing integration guide |

**Telemetry Control**

| Document | Feature | Description |
|----------|---------|-------------|
| [TELEMETRY-ENFORCEMENT.md](TELEMETRY-ENFORCEMENT.md) | Feature 1.3 | Minimum telemetry enforcement for safety-critical systems |

---

### v1.4: Security Depth (Complete ✅)

**Operational Stealth**

| Document | Feature | Description |
|----------|---------|-------------|
| [STEALTH-MODE.md](STEALTH-MODE.md) | Feature 2.1 | Low-signature execution, constant-rate timing, network fingerprint reduction |

**Threat Intelligence & Forensics**

| Document | Feature | Description |
|----------|---------|-------------|
| [THREAT-SIGNATURE.md](THREAT-SIGNATURE.md) | Feature 2.2 | Threat signature embedding for forensics and SIEM integration |

**Adversarial Testing**

| Document | Feature | Description |
|----------|---------|-------------|
| [BLUE-RED-SIMULATION.md](BLUE-RED-SIMULATION.md) | Feature 2.3 | Blue vs Red scenario simulation, dual-build testing |

**Integration**

| Document | Feature | Description |
|----------|---------|-------------|
| [V1.4-INTEGRATION-GUIDE.md](V1.4-INTEGRATION-GUIDE.md) | v1.4 Complete | Complete v1.4 integration guide combining all security features |

---

### v1.5: C3/JADC2 Operational Deployment (Complete ✅)

**JADC2 Integration & Classification Security**

| Document | Features Covered | Description |
|----------|------------------|-------------|
| [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) | 3.1, 3.2, 3.3, 3.7, 3.9 | **Complete C3/JADC2 guide** covering:<br>• Cross-domain guards & classification<br>• JADC2 & 5G/MEC integration<br>• Blue Force Tracker (BFT-2)<br>• Radio multi-protocol bridging<br>• 5G latency & throughput contracts |
| [ROADMAP-V1.5-C3-JADC2.md](ROADMAP-V1.5-C3-JADC2.md) | Planning | 11-feature C3/JADC2 roadmap and implementation phases |

**Feature 3.1**: Cross-Domain Guards & Classification
- DoD classification levels (U/C/S/TS/TS-SCI)
- Compile-time cross-domain security enforcement
- Cross-domain gateway validation
- Classification boundary metadata

**Feature 3.2**: JADC2 & 5G/Edge Integration
- 5G/MEC optimization for tactical edge nodes
- Latency budget analysis (5ms JADC2 requirement)
- Bandwidth contract enforcement (10 Gbps)
- Edge node placement recommendations

**Feature 3.3**: Blue Force Tracker (BFT-2)
- Real-time friendly force position tracking
- AES-256-GCM position encryption
- ML-DSA-87 authentication
- Spoofing detection (physical plausibility)

**Feature 3.7**: Radio Multi-Protocol Bridging
- Link-16 (tactical data link)
- SATCOM (beyond line-of-sight)
- MUOS (mobile satellite)
- SINCGARS (frequency hopping VHF)
- EPLRS (position reporting)
- Automatic jamming detection and fallback

**Feature 3.9**: 5G Latency & Throughput Contracts
- Compile-time latency verification
- URLLC (1ms ultra-reliable low-latency)
- eMBB (10 Gbps enhanced mobile broadband)
- 99.999% reliability enforcement

---

### v1.6: High-Assurance (Complete ✅) 🎉

**Nuclear Surety, Coalition Operations, & Edge Security**

| Document | Features Covered | Description |
|----------|------------------|-------------|
| [HIGH-ASSURANCE-GUIDE.md](HIGH-ASSURANCE-GUIDE.md) | 3.4, 3.5, 3.8 | **Complete high-assurance guide** covering:<br>• Two-person integrity (nuclear surety)<br>• Mission Partner Environment (MPE)<br>• Edge security hardening |

**Feature 3.4**: Two-Person Integrity for Nuclear Surety
- DOE Sigma 14 two-person integrity enforcement
- ML-DSA-87 dual-signature verification
- NC3 isolation (no network/untrusted calls)
- Nuclear Command & Control (NC3) support
- Tamper-proof audit logging (Layer 62)

**Feature 3.5**: Mission Partner Environment (MPE)
- Coalition interoperability (NATO, Five Eyes)
- Releasability controls (REL NATO, REL FVEY, NOFORN, FOUO)
- Partner validation (32 NATO + 5 FVEY nations)
- Compile-time releasability violation detection
- Runtime coalition data sharing with access control

**Feature 3.8**: Edge Security Hardening
- Hardware Security Module (HSM) integration
  - TPM 2.0 (Trusted Platform Module)
  - FIPS 140-3 Level 3 HSMs
- Secure enclave support
  - Intel SGX (Software Guard Extensions)
  - ARM TrustZone
  - AMD SEV (Secure Encrypted Virtualization)
- Remote attestation (TPM PCR measurements)
- Anti-tampering detection (physical, voltage, temperature, clock, memory, firmware)
- Emergency zeroization (DoD 5220.22-M)
- Zero-trust security model

---

## 🔧 Technical References

### Cryptography & Provenance

| Document | Description |
|----------|-------------|
| [PROVENANCE-CNSA2.md](PROVENANCE-CNSA2.md) | CNSA 2.0 provenance system with ML-DSA-87/ML-KEM-1024 |

### AI Integration

| Document | Description |
|----------|-------------|
| [AI-INTEGRATION.md](AI-INTEGRATION.md) | Layer 5/7/8 AI integration for performance, mission planning, and security |

### Enhancement Proposals

| Document | Description |
|----------|-------------|
| [ENHANCEMENT-SUGGESTIONS.md](ENHANCEMENT-SUGGESTIONS.md) | 5 strategic enhancement proposals for v1.7+ |

---

## 🎯 Quick Start by Use Case

### I want to...

**Learn the basics of DSLLVM**
1. Start with [DSLLVM-DESIGN.md](DSLLVM-DESIGN.md) - Core architecture
2. Read [ATTRIBUTES.md](ATTRIBUTES.md) - Source-level attribute reference
3. Review [PIPELINES.md](PIPELINES.md) - Compilation pipelines
4. Check [PATH-CONFIGURATION.md](PATH-CONFIGURATION.md) - Dynamic path configuration

**Build a classified military application**
1. Read [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) - Classification security
2. Review cross-domain guards (Feature 3.1)
3. Understand NOFORN, REL NATO, REL FVEY markings

**Implement JADC2 sensor fusion**
1. Read [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) - JADC2 features
2. Review 5G/MEC optimization (Feature 3.2)
3. Check latency budgets (Feature 3.9)
4. Implement BFT position tracking (Feature 3.3)

**Work with coalition partners (NATO/FVEY)**
1. Read [HIGH-ASSURANCE-GUIDE.md](HIGH-ASSURANCE-GUIDE.md) - MPE section
2. Understand releasability markings (Feature 3.5)
3. Review coalition partner lists (NATO 32, FVEY 5)
4. Check compile-time releasability enforcement

**Build nuclear weapon systems**
1. Read [HIGH-ASSURANCE-GUIDE.md](HIGH-ASSURANCE-GUIDE.md) - Nuclear surety section
2. Implement two-person integrity (Feature 3.4)
3. Ensure NC3 isolation
4. Use ML-DSA-87 signatures

**Deploy to tactical 5G edge nodes**
1. Read [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) - 5G/MEC section
2. Read [HIGH-ASSURANCE-GUIDE.md](HIGH-ASSURANCE-GUIDE.md) - Edge security section
3. Implement HSM crypto (Feature 3.8)
4. Enable remote attestation
5. Deploy tamper detection

**Build covert operations software**
1. Read [STEALTH-MODE.md](STEALTH-MODE.md) - Operational stealth
2. Enable low-signature execution
3. Use constant-rate timing
4. Suppress network fingerprints

**Integrate with Blue Force Tracker**
1. Read [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) - BFT section
2. Review BFT-2 crypto (AES-256-GCM + ML-DSA-87)
3. Implement position reporting
4. Enable spoofing detection

**Bridge tactical radios**
1. Read [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) - Radio bridging section
2. Understand Link-16, SATCOM, MUOS, SINCGARS, EPLRS
3. Implement automatic fallback
4. Enable jamming detection

---

## 📊 Feature Matrix

### Implementation Status

| Version | Phase | Features | Status | Documentation |
|---------|-------|----------|--------|---------------|
| v1.0-v1.2 | Foundation | DSMIL attributes, CNSA 2.0 provenance, AI integration | ✅ Complete | [DSLLVM-DESIGN.md](DSLLVM-DESIGN.md) |
| v1.3 | Operational Control | Mission profiles, auto-fuzzing, telemetry enforcement | ✅ Complete | [MISSION-PROFILES-GUIDE.md](MISSION-PROFILES-GUIDE.md) |
| v1.4 | Security Depth | Stealth modes, threat signatures, blue/red simulation | ✅ Complete | [STEALTH-MODE.md](STEALTH-MODE.md), [V1.4-INTEGRATION-GUIDE.md](V1.4-INTEGRATION-GUIDE.md) |
| v1.5.0 | C3/JADC2 Phase 1 | Cross-domain, JADC2, 5G/MEC | ✅ Complete | [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) |
| v1.5.1 | C3/JADC2 Phase 2 | BFT-2, radio bridging, 5G contracts | ✅ Complete | [C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md) |
| v1.6.0 | High-Assurance Phase 3 | Nuclear surety, MPE, edge security | ✅ Complete | [HIGH-ASSURANCE-GUIDE.md](HIGH-ASSURANCE-GUIDE.md) |

---

## 🔐 Security & Standards

### Military Standards Referenced

| Standard | Description | Implemented In |
|----------|-------------|----------------|
| DOE Sigma 14 | Nuclear surety two-person integrity | Feature 3.4 |
| DODI 3150.02 | DoD Nuclear Weapons Surety Program | Feature 3.4 |
| ODNI CAPCO | Controlled Access Program Coordination Office (classification) | Features 3.1, 3.5 |
| NATO STANAG 4774 | Coalition information sharing | Feature 3.5 |
| FIPS 140-3 Level 3 | Cryptographic module security | Feature 3.8 |
| TPM 2.0 | Trusted Platform Module | Feature 3.8 |
| NIST SP 800-53 | Security controls | Features 3.1, 3.8 |
| DoD 5220.22-M | Media sanitization | Feature 3.8 |

### Cryptographic Standards (CNSA 2.0)

| Algorithm | Purpose | Standard | Key/Sig Size |
|-----------|---------|----------|--------------|
| ML-DSA-87 | Post-quantum signatures | FIPS 204 | 4595-byte sig |
| ML-KEM-1024 | Post-quantum key encapsulation | FIPS 203 | 1568-byte ciphertext |
| AES-256-GCM | Symmetric encryption | FIPS 197 | 256-bit key |
| SHA3-384 | Cryptographic hashing | FIPS 202 | 384-bit hash |

---

## 🌐 Military Networks

### Supported Classification Networks

| Network | Classification | Max Level | Features |
|---------|---------------|-----------|----------|
| **NIPRNet** | UNCLASSIFIED | U | Coalition sharing, public-facing ops |
| **SIPRNet** | SECRET | U/C/S | Operational planning, intel sharing |
| **JWICS** | TOP SECRET/SCI | TS/SCI | Strategic intel, special ops |
| **NSANet** | TOP SECRET/SCI | TS/SCI | SIGINT, cryptologic ops |

### Coalition Networks

| Coalition | Nations | Releasability | Use Cases |
|-----------|---------|---------------|-----------|
| **NATO** | 32 nations | REL NATO | Alliance operations, collective defense |
| **Five Eyes (FVEY)** | 5 nations (US/UK/CA/AU/NZ) | REL FVEY | SIGINT sharing, closest allies |
| **Bilateral** | Specific partners | REL [country] | Mission-specific partnerships |

---

## 📖 Reading Order

### For New Users

1. **[DSLLVM-DESIGN.md](DSLLVM-DESIGN.md)** - Understand the architecture
2. **[ATTRIBUTES.md](ATTRIBUTES.md)** - Learn source-level attributes
3. **[PIPELINES.md](PIPELINES.md)** - Understand compilation flow
4. **Feature guides** (pick based on your use case)

### For Military Developers

1. **[C3-JADC2-INTEGRATION.md](C3-JADC2-INTEGRATION.md)** - Classification & JADC2
2. **[HIGH-ASSURANCE-GUIDE.md](HIGH-ASSURANCE-GUIDE.md)** - Nuclear, MPE, edge security
3. **[STEALTH-MODE.md](STEALTH-MODE.md)** - Covert operations
4. **[PROVENANCE-CNSA2.md](PROVENANCE-CNSA2.md)** - Supply chain security

### For Compiler Engineers

1. **[DSLLVM-DESIGN.md](DSLLVM-DESIGN.md)** - Full architecture
2. **[PIPELINES.md](PIPELINES.md)** - Pass pipelines
3. **Feature-specific passes** (C3-JADC2, HIGH-ASSURANCE)

---

## 🔄 Version History

| Version | Date | Major Changes | Documentation |
|---------|------|---------------|---------------|
| v1.0-v1.2 | 2023-2024 | Foundation, CNSA 2.0, AI integration | DSLLVM-DESIGN.md |
| v1.3 | 2024-Q2 | Mission profiles, fuzzing, telemetry | MISSION-PROFILES-GUIDE.md |
| v1.4 | 2024-Q3 | Stealth, threat signatures, blue/red | STEALTH-MODE.md |
| v1.5.0 | 2024-Q4 | Cross-domain, JADC2, 5G/MEC | C3-JADC2-INTEGRATION.md |
| v1.5.1 | 2024-Q4 | BFT-2, radio bridging, 5G contracts | C3-JADC2-INTEGRATION.md |
| v1.6.0 | 2024-Q4 | Nuclear surety, MPE, edge security | HIGH-ASSURANCE-GUIDE.md |

---

## 📞 Support & Contact

- **Project**: SWORDIntel/DSLLVM
- **Team**: DSMIL Kernel Team
- **Issues**: [GitHub Issues](https://github.com/SWORDIntel/DSLLVM/issues)
- **Documentation**: [/dsmil/docs/](/dsmil/docs/)

---

## 📝 Documentation Conventions

### File Naming

- **Uppercase with hyphens**: `FEATURE-NAME.md`
- **Version-specific**: `V1.X-FEATURE.md`
- **Integration guides**: `*-INTEGRATION.md`
- **Roadmaps**: `ROADMAP-*.md`

### Markdown Formatting

- **Headers**: Use ATX-style headers (`#`, `##`, `###`)
- **Code blocks**: Specify language for syntax highlighting
- **Tables**: Use for structured data
- **Emojis**: Used sparingly for visual organization (📚 🎯 🔧 🔐)

### Status Indicators

- ✅ **Complete**: Fully implemented and tested
- 🚧 **In Progress**: Under active development
- 📋 **Planned**: Designed but not yet implemented
- 🔬 **Research**: Experimental/research phase

---

**DSLLVM Documentation**: Comprehensive guides for the war-fighting compiler transforming military software development.
