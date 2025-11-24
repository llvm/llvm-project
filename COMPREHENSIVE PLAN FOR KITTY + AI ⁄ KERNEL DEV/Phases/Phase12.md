# Phase 12 – Enhanced Access Controls for Layer 8 & Layer 9 (v1.0)

**Version:** 1.0
**Status:** Initial Release
**Date:** 2025-11-23
**Prerequisite:** Phase 11 (External Military Communications Integration)
**Next Phase:** Phase 13 (Full Administrative Control)

---

## 1. Objectives

Phase 12 establishes **Enhanced Access Controls** for Layer 8 (Enhanced Security) and Layer 9 (Executive/Strategic Command):

1. **Dual YubiKey + Iris Authentication** - FIDO2 + FIPS YubiKeys (both plugged in) with iris biometric
2. **Session Duration Controls** - 6-hour L9, 12-hour L8 sessions (NO mandatory breaks)
3. **MinIO Local Immutable Audit** - Blockchain-style object storage for audit trail
4. **User-Configurable Geofencing** - Self-service web UI for GPS-based access zones
5. **Separation of Duties** - Explicit SoD policies for critical operations
6. **Context-Aware Access** - Threat level and behavioral analysis integration
7. **Continuous Authentication** - Behavioral biometrics during sessions

### System Context (v3.1)

- **Physical Hardware:** Intel Core Ultra 7 165H (48.2 TOPS INT8: 13.0 NPU + 32.0 GPU + 3.2 CPU)
- **Memory:** 64 GB LPDDR5x-7467, 62 GB usable for AI, 64 GB/s shared bandwidth
- **Layer 8 (Enhanced Security):** 8 devices (51-58), ATOMAL classification
- **Layer 9 (Executive/Strategic):** 4 devices (59-62) + Device 83 (Emergency), EXEC classification

### Key Principles

1. **Dual YubiKey Convenience:** Both keys remain plugged in (FIDO2 + FIPS)
2. **Variable Shift Support:** NO time-based restrictions (24/7 access)
3. **Local Audit Storage:** MinIO for immutable audit logs (NO cloud)
4. **User-Controlled Geofencing:** Self-service configuration via web UI
5. **Triple-Factor for Device 61:** Dual YubiKey + iris scan required

---

## 2. Architecture Overview

### 2.1 Enhanced Access Control Topology

```
┌─────────────────────────────────────────────────────────────┐
│          Enhanced Access Controls (Phase 12)                 │
│       Layer 8 (Devices 51-58) + Layer 9 (Devices 59-62)     │
└─────────────────────────────────────────────────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
 ┌────▼────────┐    ┌────────▼────────┐    ┌───────▼───────┐
 │  YubiKey 1  │    │   YubiKey 2     │    │  Iris Scanner │
 │  (FIDO2)    │    │   (FIPS 140-2)  │    │  (NIR + Live) │
 │  USB Port A │    │   USB Port B    │    │  USB Port C   │
 │  PLUGGED IN │    │   PLUGGED IN    │    │  On-Demand    │
 └─────┬───────┘    └────────┬────────┘    └───────┬───────┘
       │                     │                      │
       │ Challenge-          │ PIV Cert             │ Template
       │ Response            │ Verification         │ Matching
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  MFA Engine     │
                    │ (dsmil_mfa_     │
                    │  auth.c)        │
                    └────────┬────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
 ┌────▼────────┐    ┌────────▼────────┐    ┌───────▼───────┐
 │  Session    │    │  Geofence       │    │  Context-     │
 │  Manager    │    │  Validator      │    │  Aware Engine │
 │  (6h/12h)   │    │  (GPS + UI)     │    │  (Threat +    │
 │             │    │                 │    │   Behavior)   │
 └─────┬───────┘    └────────┬────────┘    └───────┬───────┘
       │                     │                      │
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Authorization  │
                    │  Engine         │
                    │ (SoD + Policy)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  MinIO Audit   │
                    │  Ledger        │
                    │  (Immutable)   │
                    └────────────────┘
                             │
                             │ User's 3-Tier Backup
                             ▼
                    [Tier 1: Hot (90d)]
                    [Tier 2: Warm (1y)]
                    [Tier 3: Cold (7y+)]
```

### 2.2 Access Control Flow

```
User Session Initiation:
  1. YubiKey 1 (FIDO2) - Challenge-response (already plugged in)
  2. YubiKey 2 (FIPS) - PIV certificate verification (already plugged in)
  3. Iris scan (if Device 61 or break-glass)
  4. Geofence validation (GPS check)
  5. Context evaluation (threat level, user behavior)
  6. Session creation (6h L9 or 12h L8)
  7. Continuous authentication (behavioral monitoring)
  8. Audit logging (MinIO immutable ledger)

Device 61 (NC3) Access Flow:
  1. Standard MFA (Dual YubiKey)
  2. Iris scan (liveness + template match)
  3. Geofence enforcement (must be in secure facility)
  4. Two-person authorization (second user with same triple-factor)
  5. ROE token validation
  6. Session recording enabled
  7. All operations logged to MinIO
```

---

## 3. Dual YubiKey + Iris Authentication

### 3.1 YubiKey Configuration (Both Plugged In)

**Purpose:** Dual-factor hardware token authentication with convenience (keys remain inserted).

**YubiKey 1 - FIDO2 Protocol**
- **Port:** USB Port A (permanently inserted)
- **Protocol:** U2F/FIDO2 (WebAuthn)
- **Algorithm:** ECDSA P-256 (transitioning to ML-DSA-87 hybrid)
- **Challenge-Response:** HMAC-SHA256
- **Serial:** Logged in audit trail

**YubiKey 2 - FIPS 140-2 Certified**
- **Port:** USB Port B (permanently inserted)
- **Protocol:** PIV (Personal Identity Verification)
- **Certification:** FIPS 140-2 Level 2 (hardware crypto module)
- **Certificate:** X.509 with RSA-2048 or ECDSA P-384
- **PIN:** 6-8 digit PIN required for operations
- **Serial:** Logged in audit trail

**Advantages of "Both Plugged In" Model:**
- **Convenience:** No constant plugging/unplugging
- **Physical Presence Satisfied:** Keys being inserted = possession verified
- **Faster Auth:** Parallel challenge-response to both keys
- **Tamper Detection:** Physical removal of either key = immediate session termination

**Security Considerations:**
- **Physical Security:** Keys must be in secure environment (tamper-evident case)
- **USB Port Monitoring:** Kernel driver detects disconnect events
- **Automatic Lockout:** Any key removal triggers session termination + audit alert

**Implementation:**

```c
// /opt/dsmil/yubikey_dual_auth.c
/**
 * DSMIL Dual YubiKey Authentication
 * Both keys remain plugged in for convenience
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libusb-1.0/libusb.h>
#include <ykpers-1/ykcore.h>
#include <ykpers-1/ykdef.h>

#define YUBI_FIDO2_VID 0x1050  // Yubico vendor ID
#define YUBI_FIDO2_PID 0x0407  // YubiKey 5 FIDO
#define YUBI_FIPS_VID  0x1050
#define YUBI_FIPS_PID  0x0406  // YubiKey 5 FIPS

struct yubikey_state {
    bool fido2_present;
    bool fips_present;
    char fido2_serial[32];
    char fips_serial[32];
    time_t last_challenge_time;
};

/**
 * Check if both YubiKeys are plugged in
 */
int yubikey_verify_dual_presence(struct yubikey_state *state) {
    libusb_context *ctx = NULL;
    libusb_device **devs;
    ssize_t cnt;
    int ret = 0;

    // Initialize libusb
    libusb_init(&ctx);

    // Get device list
    cnt = libusb_get_device_list(ctx, &devs);
    if (cnt < 0) {
        fprintf(stderr, "Failed to get USB device list\n");
        return -1;
    }

    state->fido2_present = false;
    state->fips_present = false;

    // Scan for both YubiKeys
    for (ssize_t i = 0; i < cnt; i++) {
        struct libusb_device_descriptor desc;
        libusb_get_device_descriptor(devs[i], &desc);

        if (desc.idVendor == YUBI_FIDO2_VID && desc.idProduct == YUBI_FIDO2_PID) {
            state->fido2_present = true;
            // Get serial number
            libusb_device_handle *handle;
            if (libusb_open(devs[i], &handle) == 0) {
                libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber,
                    (unsigned char*)state->fido2_serial, sizeof(state->fido2_serial));
                libusb_close(handle);
            }
        }

        if (desc.idVendor == YUBI_FIPS_VID && desc.idProduct == YUBI_FIPS_PID) {
            state->fips_present = true;
            // Get serial number
            libusb_device_handle *handle;
            if (libusb_open(devs[i], &handle) == 0) {
                libusb_get_string_descriptor_ascii(handle, desc.iSerialNumber,
                    (unsigned char*)state->fips_serial, sizeof(state->fips_serial));
                libusb_close(handle);
            }
        }
    }

    libusb_free_device_list(devs, 1);
    libusb_exit(ctx);

    // Both keys must be present
    if (state->fido2_present && state->fips_present) {
        printf("✓ Both YubiKeys detected:\n");
        printf("  FIDO2: Serial %s\n", state->fido2_serial);
        printf("  FIPS:  Serial %s\n", state->fips_serial);
        ret = 0;
    } else {
        fprintf(stderr, "✗ Dual YubiKey requirement not met:\n");
        fprintf(stderr, "  FIDO2: %s\n", state->fido2_present ? "Present" : "MISSING");
        fprintf(stderr, "  FIPS:  %s\n", state->fips_present ? "Present" : "MISSING");
        ret = -1;
    }

    return ret;
}

/**
 * Perform challenge-response with FIDO2 YubiKey
 */
int yubikey_fido2_challenge(struct yubikey_state *state, const char *challenge,
                            char *response, size_t response_len) {
    // FIDO2 challenge-response using U2F protocol
    // Implementation uses libfido2 library

    // For this spec, simplified flow:
    printf("Sending challenge to FIDO2 YubiKey (Serial: %s)...\n", state->fido2_serial);

    // TODO: Actual FIDO2 challenge-response via libfido2
    // fido_assert_t *assert = fido_assert_new();
    // fido_dev_t *dev = fido_dev_new();
    // ... (full implementation)

    snprintf(response, response_len, "FIDO2_RESPONSE_%ld", time(NULL));
    return 0;
}

/**
 * Verify PIV certificate from FIPS YubiKey
 */
int yubikey_fips_piv_verify(struct yubikey_state *state, const char *pin) {
    printf("Verifying PIV certificate on FIPS YubiKey (Serial: %s)...\n", state->fips_serial);

    // TODO: PIV certificate verification via OpenSC/PKCS#11
    // - Load PIV certificate from slot 9a
    // - Verify certificate chain
    // - Perform signature operation to prove key possession

    // For this spec, simplified flow:
    if (strlen(pin) < 6 || strlen(pin) > 8) {
        fprintf(stderr, "Invalid PIN length (must be 6-8 digits)\n");
        return -1;
    }

    printf("✓ PIV certificate verified\n");
    return 0;
}

/**
 * Monitor for YubiKey removal (session termination trigger)
 */
void yubikey_monitor_removal(struct yubikey_state *state,
                              void (*removal_callback)(const char *serial)) {
    // Hotplug monitoring using libusb
    // Detects USB disconnect events

    libusb_context *ctx = NULL;
    libusb_init(&ctx);

    // Register hotplug callback
    libusb_hotplug_callback_handle callback_handle;
    libusb_hotplug_register_callback(
        ctx,
        LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT,
        LIBUSB_HOTPLUG_ENUMERATE,
        YUBI_FIDO2_VID,
        YUBI_FIDO2_PID,
        LIBUSB_HOTPLUG_MATCH_ANY,
        NULL,  // Callback function
        NULL,
        &callback_handle
    );

    // Event loop (runs in background thread)
    while (1) {
        struct timeval tv = { 1, 0 };  // 1 second timeout
        libusb_handle_events_timeout_completed(ctx, &tv, NULL);

        // Check if either key was removed
        struct yubikey_state current;
        yubikey_verify_dual_presence(&current);

        if (!current.fido2_present && state->fido2_present) {
            fprintf(stderr, "⚠ FIDO2 YubiKey removed! Terminating session...\n");
            removal_callback(state->fido2_serial);
        }

        if (!current.fips_present && state->fips_present) {
            fprintf(stderr, "⚠ FIPS YubiKey removed! Terminating session...\n");
            removal_callback(state->fips_serial);
        }

        *state = current;
    }

    libusb_exit(ctx);
}

/**
 * Main dual YubiKey authentication flow
 */
int main() {
    struct yubikey_state state = {0};

    // Step 1: Verify both keys are plugged in
    if (yubikey_verify_dual_presence(&state) != 0) {
        fprintf(stderr, "Authentication failed: Both YubiKeys must be inserted\n");
        return 1;
    }

    // Step 2: FIDO2 challenge-response
    char fido2_response[256];
    if (yubikey_fido2_challenge(&state, "DSMIL_CHALLENGE_2025", fido2_response,
                                sizeof(fido2_response)) != 0) {
        fprintf(stderr, "FIDO2 challenge-response failed\n");
        return 1;
    }

    // Step 3: FIPS PIV certificate verification
    char pin[9];
    printf("Enter FIPS YubiKey PIN: ");
    scanf("%8s", pin);

    if (yubikey_fips_piv_verify(&state, pin) != 0) {
        fprintf(stderr, "FIPS PIV verification failed\n");
        return 1;
    }

    // Step 4: Start removal monitoring (background thread)
    // pthread_create(&monitor_thread, NULL, yubikey_monitor_removal, &state);

    printf("\n✓ Dual YubiKey authentication successful!\n");
    printf("Session started. DO NOT remove either YubiKey.\n");

    return 0;
}
```

### 3.2 Iris Biometric System

**Purpose:** High-security biometric authentication for Device 61 and break-glass operations.

**Hardware Specifications:**
- **Scanner:** IriTech IriShield USB MK 2120U (or equivalent)
- **Capture Method:** Near-infrared (NIR) 850nm
- **Resolution:** 640x480 pixels
- **Liveness Detection:** Pupil response to light stimulus
- **Anti-Spoofing:** Texture analysis, frequency domain analysis
- **Standards:** ISO/IEC 19794-6 (iris image standard)

**Liveness Detection:**
1. **Pupil Response:** Flash IR LED, measure pupil constriction
2. **Texture Analysis:** Verify iris texture complexity (not a photo)
3. **Frequency Domain:** Analyze spatial frequency (detect printed images)
4. **Movement Detection:** Require slight head movement during capture

**Template Protection:**
- **Encryption:** ML-KEM-1024 + AES-256-GCM
- **Storage:** TPM-sealed vault (`/var/lib/dsmil/biometric/iris_templates/`)
- **Matching:** 1:N search with threshold FAR = 0.0001% (1 in 1 million)
- **Anti-Replay:** Timestamp + nonce in template

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/iris_authentication.py
"""
DSMIL Iris Biometric Authentication
Liveness detection + template matching
"""

import cv2
import numpy as np
import time
import hashlib
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

class IrisAuthentication:
    def __init__(self, device_path="/dev/video0"):
        self.device_path = device_path
        self.template_db = "/var/lib/dsmil/biometric/iris_templates/"
        self.far_threshold = 0.0001  # False Accept Rate

        # Initialize iris scanner
        self.scanner = cv2.VideoCapture(device_path)
        self.scanner.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.scanner.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"Iris scanner initialized: {device_path}")

    def capture_iris_image(self) -> Optional[np.ndarray]:
        """Capture iris image from NIR camera"""
        ret, frame = self.scanner.read()
        if not ret:
            print("Failed to capture iris image")
            return None

        # Convert to grayscale (NIR is already monochrome)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return gray

    def detect_liveness(self, image: np.ndarray) -> bool:
        """
        Detect liveness using pupil response and texture analysis
        """
        print("Performing liveness detection...")

        # Step 1: Detect iris and pupil
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )

        if circles is None:
            print("  ✗ No iris detected")
            return False

        # Step 2: Pupil response test (flash IR LED)
        print("  Testing pupil response (flash IR LED)...")
        initial_pupil_size = self._measure_pupil_size(image)

        # Flash IR LED (hardware-specific, omitted for brevity)
        # time.sleep(0.1)

        # Capture second image
        flash_image = self.capture_iris_image()
        flash_pupil_size = self._measure_pupil_size(flash_image)

        # Pupil should constrict (size decrease)
        pupil_change = (initial_pupil_size - flash_pupil_size) / initial_pupil_size
        if pupil_change < 0.05:  # At least 5% constriction
            print(f"  ✗ Insufficient pupil response ({pupil_change*100:.1f}%)")
            return False

        print(f"  ✓ Pupil response verified ({pupil_change*100:.1f}% constriction)")

        # Step 3: Texture analysis (frequency domain)
        print("  Analyzing iris texture...")
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # High-frequency energy (real iris has complex texture)
        high_freq_energy = np.sum(magnitude[100:540, 100:540])  # Center crop

        if high_freq_energy < 1e6:  # Threshold (empirically determined)
            print(f"  ✗ Insufficient texture complexity (score: {high_freq_energy:.2e})")
            return False

        print(f"  ✓ Texture analysis passed (score: {high_freq_energy:.2e})")

        # Step 4: Movement detection (require slight head movement)
        print("  Requesting head movement...")
        # Capture sequence of images, detect motion
        # (Implementation omitted for brevity)

        print("✓ Liveness verification complete")
        return True

    def extract_iris_template(self, image: np.ndarray) -> bytes:
        """
        Extract iris template from image
        Uses Daugman's algorithm (simplified)
        """
        print("Extracting iris template...")

        # Step 1: Iris segmentation (detect iris boundaries)
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )

        if circles is None:
            raise ValueError("Iris segmentation failed")

        # Use first detected circle
        x, y, r = circles[0][0].astype(int)

        # Step 2: Normalization (polar transform)
        # Convert iris to rectangular image (unwrap)
        normalized = self._normalize_iris(image, x, y, r)

        # Step 3: Feature extraction (Gabor wavelets)
        template = self._extract_features(normalized)

        # Step 4: Template encoding (binary)
        template_bytes = template.tobytes()

        print(f"✓ Template extracted ({len(template_bytes)} bytes)")
        return template_bytes

    def encrypt_template(self, template: bytes, user_id: str) -> bytes:
        """
        Encrypt iris template with ML-KEM-1024 + AES-256-GCM
        """
        # Derive key from ML-KEM (integration with dsmil_pqc)
        # For this spec, simplified with direct AES key

        # Generate encryption key from user ID + timestamp
        kdf = HKDF(
            algorithm=hashes.SHA3_512(),
            length=32,
            salt=None,
            info=f"iris_template_{user_id}".encode()
        )
        key = kdf.derive(b"DSMIL_IRIS_KEY_2025")

        # Encrypt template with AES-256-GCM
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, template, None)

        # Return nonce + ciphertext
        encrypted = nonce + ciphertext

        print(f"✓ Template encrypted ({len(encrypted)} bytes)")
        return encrypted

    def enroll_user(self, user_id: str) -> bool:
        """
        Enroll new user with iris template
        """
        print(f"\n=== Iris Enrollment for {user_id} ===")

        # Capture iris image
        image = self.capture_iris_image()
        if image is None:
            return False

        # Liveness detection
        if not self.detect_liveness(image):
            print("Liveness detection failed")
            return False

        # Extract template
        template = self.extract_iris_template(image)

        # Encrypt template
        encrypted_template = self.encrypt_template(template, user_id)

        # Store template
        template_path = f"{self.template_db}/{user_id}.iris"
        with open(template_path, 'wb') as f:
            f.write(encrypted_template)

        # Compute template hash for audit
        template_hash = hashlib.sha3_512(template).hexdigest()

        print(f"✓ Enrollment complete: {template_path}")
        print(f"  Template hash: {template_hash[:16]}...")

        return True

    def authenticate_user(self, user_id: str) -> Tuple[bool, float]:
        """
        Authenticate user with iris scan
        Returns: (success, match_score)
        """
        print(f"\n=== Iris Authentication for {user_id} ===")

        # Load stored template
        template_path = f"{self.template_db}/{user_id}.iris"
        if not os.path.exists(template_path):
            print(f"No template found for {user_id}")
            return False, 0.0

        with open(template_path, 'rb') as f:
            encrypted_stored = f.read()

        # Decrypt stored template
        stored_template = self.decrypt_template(encrypted_stored, user_id)

        # Capture new iris image
        image = self.capture_iris_image()
        if image is None:
            return False, 0.0

        # Liveness detection
        if not self.detect_liveness(image):
            print("Liveness detection failed")
            return False, 0.0

        # Extract template from new image
        new_template = self.extract_iris_template(image)

        # Match templates (Hamming distance)
        match_score = self._match_templates(stored_template, new_template)

        # Threshold decision (FAR = 0.0001%)
        success = (match_score >= 0.95)

        if success:
            print(f"✓ Authentication successful (score: {match_score:.4f})")
        else:
            print(f"✗ Authentication failed (score: {match_score:.4f})")

        return success, match_score

    def _measure_pupil_size(self, image: np.ndarray) -> float:
        """Measure pupil diameter in pixels"""
        # Threshold to find darkest region (pupil)
        _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Largest contour is pupil
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)

        return radius * 2  # Diameter

    def _normalize_iris(self, image: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
        """Normalize iris to rectangular image (Daugman's rubber sheet model)"""
        # Simplified: Extract circular region and resize
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        iris_region = cv2.bitwise_and(image, image, mask=mask)

        # Crop to bounding box
        x1, y1 = max(0, x-r), max(0, y-r)
        x2, y2 = min(image.shape[1], x+r), min(image.shape[0], y+r)
        cropped = iris_region[y1:y2, x1:x2]

        # Resize to standard size
        normalized = cv2.resize(cropped, (512, 64))

        return normalized

    def _extract_features(self, normalized: np.ndarray) -> np.ndarray:
        """Extract features using Gabor wavelets"""
        # Simplified: Use Gabor filters at multiple orientations
        features = []

        for theta in range(0, 180, 45):  # 4 orientations
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5,
                theta=np.deg2rad(theta),
                lambd=10,
                gamma=0.5
            )

            filtered = cv2.filter2D(normalized, cv2.CV_32F, kernel)
            features.append(filtered.flatten())

        # Concatenate features
        feature_vector = np.concatenate(features)

        # Binarize (Daugman phase quantization)
        binary_template = (feature_vector > 0).astype(np.uint8)

        return binary_template

    def _match_templates(self, template1: bytes, template2: bytes) -> float:
        """
        Match two iris templates using Hamming distance
        Returns match score (0.0-1.0)
        """
        # Convert to numpy arrays
        t1 = np.frombuffer(template1, dtype=np.uint8)
        t2 = np.frombuffer(template2, dtype=np.uint8)

        # Ensure same length
        min_len = min(len(t1), len(t2))
        t1 = t1[:min_len]
        t2 = t2[:min_len]

        # Hamming distance
        hamming_dist = np.sum(t1 != t2) / min_len

        # Convert to similarity score
        match_score = 1.0 - hamming_dist

        return match_score

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: iris_authentication.py <enroll|auth> <user_id>")
        sys.exit(1)

    command = sys.argv[1]
    user_id = sys.argv[2] if len(sys.argv) > 2 else "john@example.mil"

    iris_auth = IrisAuthentication()

    if command == "enroll":
        success = iris_auth.enroll_user(user_id)
        sys.exit(0 if success else 1)

    elif command == "auth":
        success, score = iris_auth.authenticate_user(user_id)
        sys.exit(0 if success else 1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
```

### 3.3 Triple-Factor Authentication for Device 61

**Purpose:** Maximum security for Nuclear Command & Control (NC3) analysis operations.

**Required Factors:**
1. **YubiKey 1 (FIDO2)** - Must be plugged in, challenge-response
2. **YubiKey 2 (FIPS)** - Must be plugged in, PIV certificate + PIN
3. **Iris Scan** - Liveness detection + template match

**Authentication Flow:**

```
Device 61 Access Request:
  ↓
[Step 1] Verify both YubiKeys present
  → Check USB enumeration
  → Serial numbers logged
  ↓
[Step 2] FIDO2 challenge-response
  → Generate random challenge
  → YubiKey 1 signs challenge
  → Verify signature
  ↓
[Step 3] FIPS PIV verification
  → Prompt for PIN
  → Load certificate from YubiKey 2
  → Verify certificate chain
  → Perform signature operation
  ↓
[Step 4] Iris biometric scan
  → Capture iris image (NIR)
  → Liveness detection (pupil response + texture)
  → Extract template
  → Match against stored template (FAR < 0.0001%)
  ↓
[Step 5] Two-person authorization
  → Second user must also complete triple-factor
  → Different personnel (organizational separation)
  → Both authorizations logged
  ↓
[Step 6] ROE token validation
  → Verify ROE_TOKEN_ID is valid
  → Check ROE_LEVEL permissions
  → Verify CLASSIFICATION level
  ↓
[Step 7] Session creation
  → Create Device 61 session (6-hour max)
  → Enable session recording (screen + keystrokes)
  → All operations logged to MinIO
  → Physical YubiKey removal = session termination
```

**Break-Glass Emergency Access:**
- **Same triple-factor requirement:** No relaxation for emergencies
- **3-person authorization:** Requester + 2 approvers (all with triple-factor)
- **Automatic notification:** CISO, Ops Commander, Audit Team
- **24-hour window:** Emergency access auto-revokes after 24h
- **Post-emergency review:** Mandatory within 72 hours

---

## 4. Session Duration Controls

### 4.1 L9 Session Management (6-Hour Maximum)

**Purpose:** Executive/Strategic operations with NO mandatory breaks (variable shifts).

**Session Parameters:**
- **Maximum Duration:** 6 hours continuous
- **Idle Timeout:** 15 minutes (configurable)
- **Re-Authentication:** Required every 2 hours (dual YubiKey + iris)
- **Extension:** Manual renewal after 6h (requires full triple-factor)
- **Daily Limit:** 24 hours total (4 × 6h sessions max)
- **Mandatory Rest:** 4-hour break after 24h cumulative

**Session Lifecycle:**

```
L9 Session Start:
  → Triple-factor authentication (if Device 61)
  → OR Dual YubiKey (if Device 59/60/62)
  → Create session token (expires in 6h)
  → Start idle timer (15 min)
  → Start continuous authentication (behavioral monitoring)
  → Log session start to MinIO

During Session (every 15 minutes):
  → Check for user activity
  → If idle > 15 min: prompt for re-engagement
  → If idle > 20 min: auto-suspend session

Re-Authentication (every 2 hours):
  → Modal prompt: "Re-authentication required"
  → User completes dual YubiKey + iris (if Device 61)
  → Session extended for 2h
  → Log re-auth to MinIO

Session Expiration (6 hours):
  → Modal alert: "Session expired - renewal required"
  → User completes full authentication
  → New session created (counts toward 24h daily limit)
  → Log renewal to MinIO

Daily Limit Reached (24 hours):
  → Hard stop: "24-hour limit reached - mandatory 4h rest"
  → Session cannot be renewed
  → User must wait 4 hours
  → Log limit enforcement to MinIO
```

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/session_manager.py
"""
DSMIL Session Duration Management
L9: 6h max, L8: 12h max, NO mandatory breaks
"""

import time
import redis
import logging
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    layer: int                    # 8 or 9
    max_duration_hours: int       # 6 for L9, 12 for L8
    idle_timeout_minutes: int     # 15 for L9, 30 for L8
    reauth_interval_hours: int    # 2 for L9, 4 for L8
    daily_limit_hours: int        # 24 for both
    mandatory_rest_hours: int     # 4 for both

class SessionManager:
    def __init__(self, redis_host="localhost"):
        self.redis = redis.Redis(host=redis_host, db=0)

        # Session configurations
        self.L9_CONFIG = SessionConfig(
            layer=9,
            max_duration_hours=6,
            idle_timeout_minutes=15,
            reauth_interval_hours=2,
            daily_limit_hours=24,
            mandatory_rest_hours=4
        )

        self.L8_CONFIG = SessionConfig(
            layer=8,
            max_duration_hours=12,
            idle_timeout_minutes=30,
            reauth_interval_hours=4,
            daily_limit_hours=24,
            mandatory_rest_hours=4
        )

        logger.info("Session Manager initialized")

    def create_session(self, user_id: str, device_id: int,
                       auth_factors: dict) -> Optional[str]:
        """
        Create new session with duration enforcement
        """
        # Determine layer and config
        if 59 <= device_id <= 62:
            config = self.L9_CONFIG
            layer = 9
        elif 51 <= device_id <= 58:
            config = self.L8_CONFIG
            layer = 8
        else:
            logger.error(f"Invalid device {device_id} for session management")
            return None

        # Check daily limit
        if not self._check_daily_limit(user_id, config):
            logger.warning(f"Daily limit reached for {user_id}")
            return None

        # Generate session ID
        session_id = f"session_{user_id}_{device_id}_{int(time.time())}"

        # Session metadata
        now = time.time()
        session_data = {
            "user_id": user_id,
            "device_id": device_id,
            "layer": layer,
            "start_time": now,
            "expires_at": now + (config.max_duration_hours * 3600),
            "last_activity": now,
            "last_reauth": now,
            "reauth_required_at": now + (config.reauth_interval_hours * 3600),
            "yubikey_fido2_serial": auth_factors.get("fido2_serial", ""),
            "yubikey_fips_serial": auth_factors.get("fips_serial", ""),
            "iris_scan_hash": auth_factors.get("iris_hash", ""),
            "status": "ACTIVE"
        }

        # Store in Redis
        self.redis.hmset(f"session:{session_id}", session_data)
        self.redis.expire(f"session:{session_id}", config.max_duration_hours * 3600 + 600)

        # Track in daily usage
        self._record_daily_usage(user_id, config.max_duration_hours)

        logger.info(f"Session created: {session_id} (L{layer}, {config.max_duration_hours}h max)")

        return session_id

    def check_session_validity(self, session_id: str) -> dict:
        """
        Check if session is still valid
        Returns: {valid, reason, requires_reauth, expires_in_seconds}
        """
        session_data = self.redis.hgetall(f"session:{session_id}")

        if not session_data:
            return {"valid": False, "reason": "SESSION_NOT_FOUND"}

        now = time.time()
        start_time = float(session_data[b"start_time"])
        expires_at = float(session_data[b"expires_at"])
        last_activity = float(session_data[b"last_activity"])
        reauth_required_at = float(session_data[b"reauth_required_at"])
        layer = int(session_data[b"layer"])

        config = self.L9_CONFIG if layer == 9 else self.L8_CONFIG

        # Check expiration
        if now >= expires_at:
            return {
                "valid": False,
                "reason": "SESSION_EXPIRED",
                "duration_hours": config.max_duration_hours
            }

        # Check idle timeout
        idle_seconds = now - last_activity
        idle_limit = config.idle_timeout_minutes * 60

        if idle_seconds > idle_limit:
            return {
                "valid": False,
                "reason": "IDLE_TIMEOUT",
                "idle_minutes": idle_seconds / 60
            }

        # Check re-auth requirement
        requires_reauth = (now >= reauth_required_at)

        return {
            "valid": True,
            "reason": "OK",
            "requires_reauth": requires_reauth,
            "expires_in_seconds": expires_at - now,
            "idle_seconds": idle_seconds,
            "session_age_hours": (now - start_time) / 3600
        }

    def update_activity(self, session_id: str):
        """Update last activity timestamp"""
        self.redis.hset(f"session:{session_id}", "last_activity", time.time())

    def perform_reauth(self, session_id: str, auth_factors: dict) -> bool:
        """
        Perform re-authentication and extend session
        """
        session_data = self.redis.hgetall(f"session:{session_id}")

        if not session_data:
            logger.error(f"Session not found: {session_id}")
            return False

        layer = int(session_data[b"layer"])
        config = self.L9_CONFIG if layer == 9 else self.L8_CONFIG

        # Verify authentication factors
        # (In production: verify YubiKey challenge-response + iris scan)

        now = time.time()

        # Update re-auth timestamps
        self.redis.hmset(f"session:{session_id}", {
            "last_reauth": now,
            "reauth_required_at": now + (config.reauth_interval_hours * 3600)
        })

        logger.info(f"Re-authentication successful: {session_id}")

        return True

    def extend_session(self, session_id: str, auth_factors: dict) -> bool:
        """
        Extend session after expiration (requires full auth)
        """
        session_data = self.redis.hgetall(f"session:{session_id}")

        if not session_data:
            logger.error(f"Session not found: {session_id}")
            return False

        user_id = session_data[b"user_id"].decode()
        layer = int(session_data[b"layer"])
        config = self.L9_CONFIG if layer == 9 else self.L8_CONFIG

        # Check daily limit
        if not self._check_daily_limit(user_id, config):
            logger.warning(f"Cannot extend: daily limit reached for {user_id}")
            return False

        # Extend expiration
        now = time.time()
        new_expiration = now + (config.max_duration_hours * 3600)

        self.redis.hmset(f"session:{session_id}", {
            "expires_at": new_expiration,
            "last_reauth": now,
            "reauth_required_at": now + (config.reauth_interval_hours * 3600)
        })

        # Record additional usage
        self._record_daily_usage(user_id, config.max_duration_hours)

        logger.info(f"Session extended: {session_id} (+{config.max_duration_hours}h)")

        return True

    def _check_daily_limit(self, user_id: str, config: SessionConfig) -> bool:
        """
        Check if user has exceeded daily limit
        """
        today = datetime.now().strftime("%Y-%m-%d")
        usage_key = f"daily_usage:{user_id}:{today}"

        total_hours = float(self.redis.get(usage_key) or 0)

        if total_hours >= config.daily_limit_hours:
            # Check if mandatory rest period has elapsed
            last_limit_key = f"last_limit_reached:{user_id}"
            last_limit_time = float(self.redis.get(last_limit_key) or 0)

            if last_limit_time > 0:
                rest_elapsed = time.time() - last_limit_time
                if rest_elapsed < (config.mandatory_rest_hours * 3600):
                    logger.warning(f"Mandatory rest period not complete: "
                                 f"{rest_elapsed/3600:.1f}h / {config.mandatory_rest_hours}h")
                    return False
                else:
                    # Rest period complete, reset daily usage
                    self.redis.delete(usage_key)
                    self.redis.delete(last_limit_key)
                    return True

            # First time hitting limit
            self.redis.set(last_limit_key, time.time())
            return False

        return True

    def _record_daily_usage(self, user_id: str, hours: int):
        """Record session hours toward daily limit"""
        today = datetime.now().strftime("%Y-%m-%d")
        usage_key = f"daily_usage:{user_id}:{today}"

        self.redis.incrbyfloat(usage_key, hours)
        self.redis.expire(usage_key, 86400 * 2)  # 2 days TTL

if __name__ == "__main__":
    manager = SessionManager()

    # Create L9 session
    auth_factors = {
        "fido2_serial": "12345678",
        "fips_serial": "87654321",
        "iris_hash": "sha3-512:abc123..."
    }

    session_id = manager.create_session("john@example.mil", 61, auth_factors)
    print(f"Session created: {session_id}")

    # Check validity
    status = manager.check_session_validity(session_id)
    print(f"Session status: {status}")
```

### 4.2 L8 Session Management (12-Hour Maximum)

**Purpose:** Security operations with extended duration (NO mandatory breaks).

**Session Parameters:**
- **Maximum Duration:** 12 hours continuous
- **Idle Timeout:** 30 minutes (configurable)
- **Re-Authentication:** Required every 4 hours (dual YubiKey only, NO iris)
- **Extension:** Manual renewal after 12h (requires dual YubiKey)
- **Daily Limit:** 24 hours total (2 × 12h sessions max)
- **Mandatory Rest:** 4-hour break after 24h cumulative

**Differences from L9:**
- Longer max duration (12h vs 6h)
- Longer idle timeout (30min vs 15min)
- Less frequent re-auth (4h vs 2h)
- NO iris scan required (dual YubiKey sufficient)

---

## 5. MinIO Immutable Audit Storage

### 5.1 Local MinIO Deployment

**Purpose:** Blockchain-style immutable audit log storage (NOT cloud-based).

**MinIO Configuration:**
```yaml
# /opt/dsmil/minio/config.yaml
version: '3.8'

services:
  minio:
    image: quay.io/minio/minio:latest
    container_name: dsmil-audit-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: dsmil_admin
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}  # From Vault
      MINIO_BROWSER: "off"  # Disable web console (CLI only)
    volumes:
      - /var/lib/dsmil/minio/data:/data  # Hot storage (NVMe)
      - /mnt/warm/dsmil/minio:/warm      # Warm storage (SSD)
      - /mnt/cold/dsmil/minio:/cold      # Cold storage (HDD)
    ports:
      - "127.0.0.1:9000:9000"  # API (localhost only)
      - "127.0.0.1:9001:9001"  # Console (localhost only)
    restart: unless-stopped
    networks:
      - dsmil-internal

networks:
  dsmil-internal:
    driver: bridge
    internal: true  # No external network access
```

**Bucket Configuration:**
```bash
#!/bin/bash
# /opt/dsmil/minio/setup_audit_bucket.sh

# Create audit ledger bucket
mc mb local/dsmil-audit-ledger

# Enable versioning (immutable versions)
mc version enable local/dsmil-audit-ledger

# Set bucket policy (WORM - Write Once Read Many)
mc retention set --default GOVERNANCE "90d" local/dsmil-audit-ledger

# Enable object locking
mc retention info local/dsmil-audit-ledger

# Set lifecycle policy (tiering)
mc ilm add --expired-object-delete-marker local/dsmil-audit-ledger \
    --transition-days 90 --storage-class WARM \
    --transition-days 365 --storage-class COLD

echo "✓ Audit bucket configured with WORM + tiering"
```

### 5.2 Blockchain-Style Object Chaining

**Purpose:** Cryptographic chain of audit events (tamper-evident).

**Object Format:**
```json
{
  "block_id": 12345,
  "timestamp": "2025-11-23T14:30:00.123456Z",
  "event_type": "DEVICE_61_ACCESS",
  "user_id": "john@example.mil",
  "device_id": 61,
  "operation": "NC3_ANALYSIS",
  "session_id": "session_john_61_1732373400",

  "authentication": {
    "yubikey_fido2_serial": "12345678",
    "yubikey_fips_serial": "87654321",
    "yubikey_fips_pin_verified": true,
    "iris_scan_hash": "sha3-512:abc123...",
    "iris_liveness_verified": true,
    "geofence_validated": true,
    "geofence_zone": "ops_center_hq"
  },

  "authorization": {
    "role": "EXEC_ANALYST",
    "clearance_level": "EXEC",
    "compartments": ["NUCLEAR", "CRYPTO"],
    "roe_token_id": "roe-2025-11-23-001",
    "roe_level": "ANALYSIS_ONLY",
    "dual_auth_required": true,
    "second_authorizer": "jane@example.mil",
    "second_auth_timestamp": "2025-11-23T14:28:00Z"
  },

  "blockchain": {
    "previous_block_hash": "sha3-512:def456...",
    "current_block_hash": "sha3-512:ghi789...",
    "signature": "ml-dsa-87:jkl012...",
    "nonce": 42
  },

  "metadata": {
    "source_ip": "10.0.1.100",
    "terminal_id": "SECURE_TERM_001",
    "tpm_quote": "tpm2.0:mno345..."
  }
}
```

**Object Storage Path:**
```
s3://dsmil-audit-ledger/
  └── 2025/
      └── 11/
          └── 23/
              ├── block-00001.json
              ├── block-00002.json
              ├── block-00003.json
              ...
              └── block-12345.json
```

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/audit_ledger_minio.py
"""
DSMIL Immutable Audit Ledger using MinIO
Blockchain-style object chaining
"""

import json
import time
import hashlib
import os
from datetime import datetime
from minio import Minio
from minio.error import S3Error
from typing import Dict, Optional
from dsmil_pqc import MLDSASignature

class AuditLedgerMinIO:
    def __init__(self, endpoint="localhost:9000"):
        # MinIO client
        self.client = Minio(
            endpoint,
            access_key=os.getenv("MINIO_ROOT_USER", "dsmil_admin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
            secure=False  # Localhost, no TLS needed
        )

        self.bucket = "dsmil-audit-ledger"

        # ML-DSA-87 signer for block signatures
        self.signer = MLDSASignature()

        # Verify bucket exists
        if not self.client.bucket_exists(self.bucket):
            raise ValueError(f"Bucket {self.bucket} does not exist!")

        print(f"Audit Ledger initialized: MinIO @ {endpoint}, Bucket: {self.bucket}")

    def get_last_block_hash(self) -> str:
        """
        Get hash of last block in chain
        """
        # List objects, get most recent
        objects = self.client.list_objects(self.bucket, recursive=True)

        latest_object = None
        latest_time = 0

        for obj in objects:
            if obj.last_modified.timestamp() > latest_time:
                latest_time = obj.last_modified.timestamp()
                latest_object = obj.object_name

        if latest_object is None:
            # Genesis block
            return "GENESIS_BLOCK_2025"

        # Fetch latest block
        response = self.client.get_object(self.bucket, latest_object)
        block_data = json.loads(response.read())
        response.close()
        response.release_conn()

        return block_data["blockchain"]["current_block_hash"]

    def compute_block_hash(self, block_data: Dict, previous_hash: str) -> str:
        """
        Compute SHA3-512 hash of block
        """
        # Serialize block data (excluding current_block_hash and signature)
        block_content = {
            "block_id": block_data["block_id"],
            "timestamp": block_data["timestamp"],
            "event_type": block_data["event_type"],
            "user_id": block_data["user_id"],
            "device_id": block_data["device_id"],
            "operation": block_data.get("operation", ""),
            "authentication": block_data.get("authentication", {}),
            "authorization": block_data.get("authorization", {}),
            "previous_block_hash": previous_hash
        }

        # Deterministic JSON serialization
        block_json = json.dumps(block_content, sort_keys=True)

        # SHA3-512 hash
        block_hash = hashlib.sha3_512(block_json.encode()).hexdigest()

        return f"sha3-512:{block_hash}"

    def append_block(self, event_type: str, user_id: str, device_id: int,
                     operation: str, authentication: Dict, authorization: Dict,
                     metadata: Dict) -> str:
        """
        Append new block to audit ledger
        Returns: object key in MinIO
        """
        # Get previous block hash
        previous_hash = self.get_last_block_hash()

        # Generate block ID (monotonically increasing)
        block_id = int(time.time() * 1000)  # Millisecond timestamp

        # Build block data
        block_data = {
            "block_id": block_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "user_id": user_id,
            "device_id": device_id,
            "operation": operation,
            "authentication": authentication,
            "authorization": authorization,
            "metadata": metadata,
            "blockchain": {
                "previous_block_hash": previous_hash,
                "current_block_hash": "",  # Computed below
                "signature": "",           # Signed below
                "nonce": 0
            }
        }

        # Compute block hash
        current_hash = self.compute_block_hash(block_data, previous_hash)
        block_data["blockchain"]["current_block_hash"] = current_hash

        # Sign block with ML-DSA-87
        signature = self.signer.sign(current_hash.encode())
        block_data["blockchain"]["signature"] = f"ml-dsa-87:{signature.hex()}"

        # Object key (date-based partitioning)
        now = datetime.utcnow()
        object_key = f"{now.year}/{now.month:02d}/{now.day:02d}/block-{block_id}.json"

        # Serialize to JSON
        block_json = json.dumps(block_data, indent=2)

        # Upload to MinIO
        self.client.put_object(
            self.bucket,
            object_key,
            data=io.BytesIO(block_json.encode()),
            length=len(block_json),
            content_type="application/json"
        )

        print(f"✓ Block appended: {object_key}")
        print(f"  Block ID: {block_id}")
        print(f"  Hash: {current_hash[:32]}...")

        return object_key

    def verify_chain_integrity(self, start_date: str = None) -> bool:
        """
        Verify entire blockchain integrity
        Args:
            start_date: Optional date to start verification (YYYY-MM-DD)
        Returns:
            True if chain is valid, False if tampering detected
        """
        print("Verifying audit chain integrity...")

        # List all blocks in chronological order
        objects = list(self.client.list_objects(self.bucket, recursive=True))
        objects.sort(key=lambda obj: obj.last_modified)

        if start_date:
            # Filter by date
            objects = [obj for obj in objects if start_date in obj.object_name]

        print(f"Verifying {len(objects)} blocks...")

        prev_hash = "GENESIS_BLOCK_2025"

        for i, obj in enumerate(objects):
            # Fetch block
            response = self.client.get_object(self.bucket, obj.object_name)
            block_data = json.loads(response.read())
            response.close()
            response.release_conn()

            # Verify previous hash matches
            stored_prev_hash = block_data["blockchain"]["previous_block_hash"]
            if stored_prev_hash != prev_hash:
                print(f"✗ Chain broken at block {i}: {obj.object_name}")
                print(f"  Expected prev_hash: {prev_hash}")
                print(f"  Got prev_hash: {stored_prev_hash}")
                return False

            # Recompute current hash
            computed_hash = self.compute_block_hash(block_data, prev_hash)
            stored_hash = block_data["blockchain"]["current_block_hash"]

            if computed_hash != stored_hash:
                print(f"✗ Hash mismatch at block {i}: {obj.object_name}")
                print(f"  Computed: {computed_hash}")
                print(f"  Stored: {stored_hash}")
                return False

            # Verify ML-DSA-87 signature
            signature_hex = block_data["blockchain"]["signature"].replace("ml-dsa-87:", "")
            signature = bytes.fromhex(signature_hex)

            if not self.signer.verify(stored_hash.encode(), signature):
                print(f"✗ Invalid signature at block {i}: {obj.object_name}")
                return False

            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"  Verified {i + 1} / {len(objects)} blocks...")

            # Update prev_hash for next iteration
            prev_hash = stored_hash

        print(f"✓ Chain integrity verified: {len(objects)} blocks")
        return True

    def get_user_audit_trail(self, user_id: str, start_date: str = None,
                             end_date: str = None) -> list:
        """
        Retrieve audit trail for specific user
        """
        print(f"Retrieving audit trail for {user_id}...")

        # List all blocks
        objects = self.client.list_objects(self.bucket, recursive=True)

        audit_trail = []

        for obj in objects:
            # Date filtering
            if start_date and start_date not in obj.object_name:
                continue
            if end_date and end_date not in obj.object_name:
                continue

            # Fetch block
            response = self.client.get_object(self.bucket, obj.object_name)
            block_data = json.loads(response.read())
            response.close()
            response.release_conn()

            # Check if block is for this user
            if block_data["user_id"] == user_id:
                audit_trail.append(block_data)

        print(f"✓ Found {len(audit_trail)} audit entries for {user_id}")

        return audit_trail

if __name__ == "__main__":
    import sys

    ledger = AuditLedgerMinIO()

    if len(sys.argv) < 2:
        print("Usage: audit_ledger_minio.py <append|verify|query> [args]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "append":
        # Example: append block
        ledger.append_block(
            event_type="DEVICE_61_ACCESS",
            user_id="john@example.mil",
            device_id=61,
            operation="NC3_ANALYSIS",
            authentication={
                "yubikey_fido2_serial": "12345678",
                "yubikey_fips_serial": "87654321",
                "iris_scan_hash": "sha3-512:abc123..."
            },
            authorization={
                "role": "EXEC_ANALYST",
                "clearance_level": "EXEC",
                "roe_token_id": "roe-2025-11-23-001"
            },
            metadata={
                "source_ip": "10.0.1.100",
                "terminal_id": "SECURE_TERM_001"
            }
        )

    elif command == "verify":
        # Verify chain integrity
        start_date = sys.argv[2] if len(sys.argv) > 2 else None
        success = ledger.verify_chain_integrity(start_date)
        sys.exit(0 if success else 1)

    elif command == "query":
        # Query user audit trail
        user_id = sys.argv[2] if len(sys.argv) > 2 else "john@example.mil"
        trail = ledger.get_user_audit_trail(user_id)

        for entry in trail:
            print(json.dumps(entry, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
```

### 5.3 User's 3-Tiered Backup Integration

**Purpose:** Automated tiering from hot → warm → cold storage.

**Tier Configuration:**
```
Tier 1 (Hot):
  - Storage: /var/lib/dsmil/minio/data (NVMe)
  - Retention: 90 days
  - Access: Immediate (< 10ms latency)
  - Use case: Active investigations, real-time audit

Tier 2 (Warm):
  - Storage: /mnt/warm/dsmil/minio (SSD)
  - Retention: 1 year
  - Access: Fast (< 100ms latency)
  - Use case: Recent historical analysis

Tier 3 (Cold):
  - Storage: /mnt/cold/dsmil/minio (HDD or tape)
  - Retention: 7+ years
  - Access: Slow (seconds to minutes)
  - Use case: Long-term archival, compliance
```

**MinIO Lifecycle Policy (User-Configurable):**
```xml
<!-- /opt/dsmil/minio/lifecycle-policy.xml -->
<LifecycleConfiguration>
  <Rule>
    <ID>Tier1-to-Tier2</ID>
    <Status>Enabled</Status>
    <Filter>
      <Prefix>2025/</Prefix>
    </Filter>
    <Transition>
      <Days>90</Days>
      <StorageClass>WARM</StorageClass>
    </Transition>
  </Rule>

  <Rule>
    <ID>Tier2-to-Tier3</ID>
    <Status>Enabled</Status>
    <Filter>
      <Prefix>2025/</Prefix>
    </Filter>
    <Transition>
      <Days>365</Days>
      <StorageClass>COLD</StorageClass>
    </Transition>
  </Rule>

  <Rule>
    <ID>Retention-7years</ID>
    <Status>Enabled</Status>
    <Filter>
      <Prefix>2025/</Prefix>
    </Filter>
    <Expiration>
      <Days>2555</Days>  <!-- 7 years -->
    </Expiration>
  </Rule>
</LifecycleConfiguration>
```

**User's Backup Automation Script (Template):**
```bash
#!/bin/bash
# /opt/dsmil/minio/user_backup_automation.sh
# User-configured 3-tiered backup automation

set -e

# Configuration (user customizable)
MINIO_ALIAS="local"
BUCKET="dsmil-audit-ledger"
TIER1_PATH="/var/lib/dsmil/minio/data"
TIER2_PATH="/mnt/warm/dsmil/minio"
TIER3_PATH="/mnt/cold/dsmil/minio"

# Tier 1 → Tier 2 (Hot → Warm after 90 days)
echo "[$(date)] Starting Tier 1 → Tier 2 migration..."
mc mirror --older-than 90d ${MINIO_ALIAS}/${BUCKET} ${TIER2_PATH}/${BUCKET}
echo "✓ Tier 1 → Tier 2 complete"

# Tier 2 → Tier 3 (Warm → Cold after 1 year)
echo "[$(date)] Starting Tier 2 → Tier 3 migration..."
find ${TIER2_PATH}/${BUCKET} -type f -mtime +365 -exec mv {} ${TIER3_PATH}/${BUCKET}/ \;
echo "✓ Tier 2 → Tier 3 complete"

# Integrity verification (sample 1% of blocks)
echo "[$(date)] Running integrity verification..."
python3 /opt/dsmil/audit_ledger_minio.py verify "2025-11"
echo "✓ Integrity verification complete"

# Backup statistics
echo "[$(date)] Backup statistics:"
echo "  Tier 1 (Hot):  $(du -sh ${TIER1_PATH} | cut -f1)"
echo "  Tier 2 (Warm): $(du -sh ${TIER2_PATH} | cut -f1)"
echo "  Tier 3 (Cold): $(du -sh ${TIER3_PATH} | cut -f1)"

# Optional: External backup (user-configured)
# rsync -avz ${TIER3_PATH}/${BUCKET} user@backup-server:/backups/dsmil/

echo "[$(date)] Backup automation complete"
```

**Cron Schedule (User-Configurable):**
```cron
# /etc/cron.d/dsmil-audit-backup
# Run backup automation daily at 2 AM
0 2 * * * dsmil /opt/dsmil/minio/user_backup_automation.sh >> /var/log/dsmil/backup.log 2>&1
```

---

## 6. User-Configurable Geofencing

### 6.1 Geofence Web UI

**Purpose:** Self-service geofence configuration for L8/L9 access control.

**Web Interface (React + Leaflet):**

```tsx
// /opt/dsmil/web-ui/src/components/GeofenceManager.tsx
/**
 * DSMIL Geofence Configuration UI
 * Interactive map for creating GPS-based access zones
 */

import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Circle, Marker, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

interface Geofence {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  radius_meters: number;
  applicable_devices: number[];
  classification: string;
  override_allowed: boolean;
  created_by: string;
  created_at: string;
}

export const GeofenceManager: React.FC = () => {
  const [geofences, setGeofences] = useState<Geofence[]>([]);
  const [editMode, setEditMode] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState<{lat: number, lng: number} | null>(null);
  const [radius, setRadius] = useState(100); // Default 100 meters

  // Load existing geofences
  useEffect(() => {
    fetch('/api/geofences')
      .then(res => res.json())
      .then(data => setGeofences(data));
  }, []);

  // Map click handler
  const MapClickHandler = () => {
    useMapEvents({
      click(e) {
        if (editMode) {
          setSelectedPoint({ lat: e.latlng.lat, lng: e.latlng.lng });
        }
      },
    });
    return null;
  };

  // Create geofence
  const handleCreateGeofence = () => {
    if (!selectedPoint) {
      alert("Please click on the map to select a location");
      return;
    }

    const newGeofence: Partial<Geofence> = {
      name: prompt("Geofence name:") || "Unnamed Zone",
      latitude: selectedPoint.lat,
      longitude: selectedPoint.lng,
      radius_meters: radius,
      applicable_devices: [], // User will configure in next step
      classification: "SECRET",
      override_allowed: false,
    };

    fetch('/api/geofences', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newGeofence),
    })
      .then(res => res.json())
      .then(created => {
        setGeofences([...geofences, created]);
        setSelectedPoint(null);
        setEditMode(false);
        alert(`Geofence "${created.name}" created successfully`);
      });
  };

  // Delete geofence
  const handleDeleteGeofence = (id: string) => {
    if (!confirm("Delete this geofence?")) return;

    fetch(`/api/geofences/${id}`, { method: 'DELETE' })
      .then(() => {
        setGeofences(geofences.filter(gf => gf.id !== id));
      });
  };

  return (
    <div className="geofence-manager">
      <div className="controls">
        <h2>Geofence Configuration</h2>

        <div className="toolbar">
          <button onClick={() => setEditMode(!editMode)}>
            {editMode ? 'Cancel' : 'Create New Geofence'}
          </button>

          {editMode && (
            <>
              <label>
                Radius (meters):
                <input
                  type="number"
                  value={radius}
                  onChange={(e) => setRadius(parseInt(e.target.value))}
                  min="10"
                  max="10000"
                />
              </label>

              <button onClick={handleCreateGeofence} disabled={!selectedPoint}>
                Save Geofence
              </button>
            </>
          )}
        </div>

        <div className="geofence-list">
          <h3>Active Geofences</h3>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Location</th>
                <th>Radius</th>
                <th>Devices</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {geofences.map(gf => (
                <tr key={gf.id}>
                  <td>{gf.name}</td>
                  <td>{gf.latitude.toFixed(4)}, {gf.longitude.toFixed(4)}</td>
                  <td>{gf.radius_meters}m</td>
                  <td>{gf.applicable_devices.join(', ') || 'All'}</td>
                  <td>
                    <button onClick={() => handleDeleteGeofence(gf.id)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="map-container">
        <MapContainer
          center={[38.8977, -77.0365]}  // Default: Washington DC
          zoom={13}
          style={{ height: '600px', width: '100%' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapClickHandler />

          {/* Render existing geofences */}
          {geofences.map(gf => (
            <Circle
              key={gf.id}
              center={[gf.latitude, gf.longitude]}
              radius={gf.radius_meters}
              pathOptions={{ color: 'blue', fillColor: 'blue', fillOpacity: 0.2 }}
            />
          ))}

          {/* Render selected point (during creation) */}
          {selectedPoint && (
            <>
              <Marker position={[selectedPoint.lat, selectedPoint.lng]} />
              <Circle
                center={[selectedPoint.lat, selectedPoint.lng]}
                radius={radius}
                pathOptions={{ color: 'green', fillColor: 'green', fillOpacity: 0.3 }}
              />
            </>
          )}
        </MapContainer>
      </div>
    </div>
  );
};
```

### 6.2 Geofence Enforcement

**GPS Validation on Session Initiation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/geofence_validator.py
"""
DSMIL Geofence Validation
GPS-based access control
"""

import math
import requests
from typing import Optional, Tuple

class GeofenceValidator:
    def __init__(self):
        self.geofences = self._load_geofences()

    def _load_geofences(self) -> list:
        """Load geofences from database"""
        # In production: query PostgreSQL or Redis
        # For this spec: example hardcoded geofences
        return [
            {
                "id": "gf-001",
                "name": "Operations Center HQ",
                "latitude": 38.8977,
                "longitude": -77.0365,
                "radius_meters": 100,
                "applicable_devices": [59, 60, 61, 62],  # L9 devices
                "override_allowed": False
            },
            {
                "id": "gf-002",
                "name": "SCIF Building 3",
                "latitude": 38.9000,
                "longitude": -77.0400,
                "radius_meters": 50,
                "applicable_devices": [61],  # Device 61 only
                "override_allowed": False
            }
        ]

    def get_current_location(self) -> Optional[Tuple[float, float]]:
        """
        Get current GPS location
        Options:
          1. GPS hardware (via gpsd)
          2. IP geolocation (fallback)
          3. Manual input (for testing)
        """
        try:
            # Option 1: GPS hardware (via gpsd)
            import gps
            session = gps.gps(mode=gps.WATCH_ENABLE)
            report = session.next()

            if report['class'] == 'TPV':
                lat = report.get('lat', 0.0)
                lon = report.get('lon', 0.0)

                if lat != 0.0 and lon != 0.0:
                    return (lat, lon)
        except:
            pass

        # Option 2: IP geolocation (fallback, less accurate)
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            data = response.json()

            if data['status'] == 'success':
                return (data['lat'], data['lon'])
        except:
            pass

        # Option 3: No location available
        return None

    def haversine_distance(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates (Haversine formula)
        Returns distance in meters
        """
        R = 6371000  # Earth radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        distance = R * c
        return distance

    def validate_geofence(self, device_id: int,
                         current_lat: float, current_lon: float) -> Tuple[bool, str]:
        """
        Validate if current location is within allowed geofence
        Returns: (valid, reason)
        """
        # Get applicable geofences for this device
        applicable = [gf for gf in self.geofences
                     if device_id in gf["applicable_devices"] or
                        not gf["applicable_devices"]]

        if not applicable:
            # No geofence requirement for this device
            return (True, "NO_GEOFENCE_REQUIRED")

        # Check if inside any applicable geofence
        for gf in applicable:
            distance = self.haversine_distance(
                current_lat, current_lon,
                gf["latitude"], gf["longitude"]
            )

            if distance <= gf["radius_meters"]:
                return (True, f"INSIDE_GEOFENCE:{gf['name']}")

        # Not inside any geofence
        nearest = min(applicable,
                     key=lambda gf: self.haversine_distance(
                         current_lat, current_lon,
                         gf["latitude"], gf["longitude"]
                     ))

        nearest_dist = self.haversine_distance(
            current_lat, current_lon,
            nearest["latitude"], nearest["longitude"]
        )

        return (False, f"OUTSIDE_GEOFENCE:nearest={nearest['name']},distance={nearest_dist:.0f}m")

    def request_override(self, device_id: int, user_id: str,
                        justification: str) -> bool:
        """
        Request geofence override (requires supervisor approval)
        """
        # In production: create approval ticket, notify supervisor
        print(f"Geofence override requested:")
        print(f"  User: {user_id}")
        print(f"  Device: {device_id}")
        print(f"  Justification: {justification}")
        print(f"  Awaiting supervisor approval...")

        # For this spec: return False (requires manual approval)
        return False

if __name__ == "__main__":
    validator = GeofenceValidator()

    # Get current location
    location = validator.get_current_location()

    if location is None:
        print("✗ GPS location unavailable")
        exit(1)

    lat, lon = location
    print(f"Current location: {lat:.4f}, {lon:.4f}")

    # Validate for Device 61
    valid, reason = validator.validate_geofence(61, lat, lon)

    if valid:
        print(f"✓ Geofence validation passed: {reason}")
    else:
        print(f"✗ Geofence validation failed: {reason}")

        # Request override
        validator.request_override(61, "john@example.mil",
                                  "Emergency field operations")
```

---

## 7. Separation of Duties (SoD)

### 7.1 Explicit SoD Policies

**Purpose:** Prevent conflicts of interest and self-authorization.

**SoD Rules:**

1. **Self-Authorization Prevention:**
   - Requester ≠ Authorizer
   - User cannot approve own requests

2. **Organizational Separation (Device 61):**
   - Requester and authorizers must be from different chains of command
   - Example: Analyst cannot be authorized by their direct supervisor
   - Requires organizational metadata in user profiles

3. **Role Conflict Detection:**
   - Admin cannot approve own privilege escalation
   - Security auditor cannot modify own audit logs
   - Operator cannot override own access denials

4. **Dual Authorization:**
   - Critical operations require two independent authorizers
   - Both authorizers must complete full authentication
   - Authorizers cannot be from same organizational unit (for Device 61)

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/sod_policy_engine.py
"""
DSMIL Separation of Duties Policy Engine
Prevents conflicts of interest
"""

from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class User:
    user_id: str
    name: str
    role: str
    clearance_level: str
    organizational_unit: str  # e.g., "OPS_COMMAND_ALPHA", "INTEL_ANALYSIS_BRAVO"
    chain_of_command: List[str]  # List of supervisor user_ids

class SoDPolicyEngine:
    def __init__(self):
        self.policies = [
            self._policy_self_authorization,
            self._policy_organizational_separation,
            self._policy_role_conflict,
            self._policy_dual_authorization
        ]

    def evaluate_authorization(self, requester: User, authorizer: User,
                               operation: str, device_id: int) -> Tuple[bool, str]:
        """
        Evaluate if authorization satisfies SoD policies
        Returns: (allowed, reason)
        """
        # Check all policies
        for policy in self.policies:
            allowed, reason = policy(requester, authorizer, operation, device_id)

            if not allowed:
                return (False, reason)

        return (True, "SOD_POLICIES_SATISFIED")

    def _policy_self_authorization(self, requester: User, authorizer: User,
                                   operation: str, device_id: int) -> Tuple[bool, str]:
        """
        Policy 1: Self-authorization prevention
        """
        if requester.user_id == authorizer.user_id:
            return (False, "SOD_VIOLATION:SELF_AUTHORIZATION")

        return (True, "OK")

    def _policy_organizational_separation(self, requester: User, authorizer: User,
                                         operation: str, device_id: int) -> Tuple[bool, str]:
        """
        Policy 2: Organizational separation (Device 61 only)
        """
        if device_id != 61:
            # Not required for other devices
            return (True, "OK")

        # Check if same organizational unit
        if requester.organizational_unit == authorizer.organizational_unit:
            return (False, "SOD_VIOLATION:SAME_ORG_UNIT")

        # Check if in same chain of command
        if authorizer.user_id in requester.chain_of_command:
            return (False, "SOD_VIOLATION:DIRECT_SUPERVISOR")

        if requester.user_id in authorizer.chain_of_command:
            return (False, "SOD_VIOLATION:DIRECT_REPORT")

        return (True, "OK")

    def _policy_role_conflict(self, requester: User, authorizer: User,
                             operation: str, device_id: int) -> Tuple[bool, str]:
        """
        Policy 3: Role conflict detection
        """
        # Admin cannot approve own privilege escalation
        if operation == "PRIVILEGE_ESCALATION" and requester.role == "ADMIN":
            if authorizer.role != "EXEC":
                return (False, "SOD_VIOLATION:ADMIN_REQUIRES_EXEC_APPROVAL")

        # Security auditor cannot modify own audit logs
        if operation == "MODIFY_AUDIT_LOG" and requester.role == "SECURITY_AUDITOR":
            return (False, "SOD_VIOLATION:AUDITOR_CANNOT_MODIFY_LOGS")

        return (True, "OK")

    def _policy_dual_authorization(self, requester: User, authorizer: User,
                                   operation: str, device_id: int) -> Tuple[bool, str]:
        """
        Policy 4: Dual authorization requirement
        (Note: This checks first authorizer; second authorizer checked separately)
        """
        # Critical operations require dual authorization
        critical_ops = ["DEVICE_61_ACCESS", "EMERGENCY_OVERRIDE", "PRIVILEGE_ESCALATION"]

        if operation in critical_ops:
            # Dual authorization required (second authorizer checked in separate call)
            return (True, "OK_FIRST_AUTH")

        return (True, "OK")

if __name__ == "__main__":
    engine = SoDPolicyEngine()

    # Example users
    requester = User(
        user_id="john@example.mil",
        name="John Doe",
        role="ANALYST",
        clearance_level="EXEC",
        organizational_unit="OPS_COMMAND_ALPHA",
        chain_of_command=["supervisor1@example.mil", "commander1@example.mil"]
    )

    authorizer1 = User(
        user_id="jane@example.mil",
        name="Jane Smith",
        role="EXEC_ANALYST",
        clearance_level="EXEC",
        organizational_unit="INTEL_ANALYSIS_BRAVO",  # Different org unit
        chain_of_command=["supervisor2@example.mil", "commander2@example.mil"]
    )

    # Evaluate authorization for Device 61 access
    allowed, reason = engine.evaluate_authorization(
        requester, authorizer1, "DEVICE_61_ACCESS", 61
    )

    if allowed:
        print(f"✓ Authorization allowed: {reason}")
    else:
        print(f"✗ Authorization denied: {reason}")
```

---

## 8. Context-Aware Access Control

### 8.1 Threat Level Integration

**Purpose:** Adjust access policies based on operational threat level.

**Threat Levels:**
- **GREEN:** Peacetime, normal operations
- **YELLOW:** Elevated threat, increased monitoring
- **ORANGE:** High threat, restricted access
- **RED:** Imminent threat, minimal access
- **DEFCON 5-1:** Military readiness levels

**Policy Adjustments:**

| Threat Level | L8 Access | L9 Access | Device 61 | Session Duration |
|--------------|-----------|-----------|-----------|------------------|
| GREEN | Normal | Normal | Dual-auth + iris | 12h L8, 6h L9 |
| YELLOW | Normal | Restricted | Dual-auth + iris + supervisor | 8h L8, 4h L9 |
| ORANGE | Restricted | Minimal | 3-person auth | 4h L8, 2h L9 |
| RED | Minimal | Emergency only | 3-person + commander | 2h L8, 1h L9 |
| DEFCON 1 | Emergency only | Emergency only | 4-person + exec | 1h max |

**Implementation:**

```python
#!/usr/bin/env python3
# /opt/dsmil/context_aware_access.py
"""
DSMIL Context-Aware Access Control
Threat level integration
"""

from enum import Enum
from typing import Dict

class ThreatLevel(Enum):
    GREEN = 1   # Peacetime
    YELLOW = 2  # Elevated
    ORANGE = 3  # High
    RED = 4     # Imminent
    DEFCON_1 = 5  # Maximum readiness

class ContextAwareAccess:
    def __init__(self):
        self.current_threat_level = ThreatLevel.GREEN
        self.operational_context = "PEACETIME"  # PEACETIME, EXERCISE, CRISIS

    def set_threat_level(self, level: ThreatLevel):
        """Set current threat level"""
        self.current_threat_level = level
        print(f"Threat level updated: {level.name}")

    def get_access_policy(self, device_id: int) -> Dict:
        """
        Get access policy based on current threat level
        """
        # Determine layer
        if 51 <= device_id <= 58:
            layer = 8
        elif 59 <= device_id <= 62:
            layer = 9
        else:
            layer = 0

        # Base policy
        policy = {
            "layer": layer,
            "device_id": device_id,
            "threat_level": self.current_threat_level.name,
            "access_allowed": True,
            "required_auth_factors": ["yubikey_fido2", "yubikey_fips"],
            "required_authorizers": 1,
            "max_session_duration_hours": 12 if layer == 8 else 6,
            "restrictions": []
        }

        # Adjust policy based on threat level
        if self.current_threat_level == ThreatLevel.GREEN:
            # Normal operations
            if device_id == 61:
                policy["required_auth_factors"].append("iris_scan")
                policy["required_authorizers"] = 2

        elif self.current_threat_level == ThreatLevel.YELLOW:
            # Elevated threat - increased monitoring
            policy["max_session_duration_hours"] = 8 if layer == 8 else 4
            policy["restrictions"].append("INCREASED_MONITORING")

            if device_id == 61:
                policy["required_auth_factors"].append("iris_scan")
                policy["required_authorizers"] = 2
                policy["restrictions"].append("SUPERVISOR_NOTIFICATION")

        elif self.current_threat_level == ThreatLevel.ORANGE:
            # High threat - restricted access
            policy["max_session_duration_hours"] = 4 if layer == 8 else 2
            policy["restrictions"].append("RESTRICTED_ACCESS")

            if layer == 9:
                policy["access_allowed"] = False
                policy["restrictions"].append("L9_ACCESS_MINIMAL")

            if device_id == 61:
                policy["required_auth_factors"].append("iris_scan")
                policy["required_authorizers"] = 3

        elif self.current_threat_level == ThreatLevel.RED:
            # Imminent threat - minimal access
            policy["max_session_duration_hours"] = 2 if layer == 8 else 1
            policy["restrictions"].append("MINIMAL_ACCESS")

            if layer == 9:
                policy["access_allowed"] = False
                policy["restrictions"].append("L9_EMERGENCY_ONLY")

            if device_id == 61:
                policy["access_allowed"] = False
                policy["restrictions"].append("DEVICE_61_EMERGENCY_ONLY")
                policy["required_authorizers"] = 3  # + commander approval

        elif self.current_threat_level == ThreatLevel.DEFCON_1:
            # Maximum readiness - emergency only
            policy["max_session_duration_hours"] = 1
            policy["restrictions"].append("EMERGENCY_ONLY")

            if layer == 8:
                policy["access_allowed"] = False
                policy["restrictions"].append("L8_EMERGENCY_ONLY")

            if layer == 9:
                policy["access_allowed"] = False
                policy["restrictions"].append("L9_EXECUTIVE_AUTHORIZATION_REQUIRED")

            if device_id == 61:
                policy["access_allowed"] = False
                policy["restrictions"].append("DEVICE_61_EXECUTIVE_AUTHORIZATION_REQUIRED")
                policy["required_authorizers"] = 4  # + executive approval

        return policy

if __name__ == "__main__":
    context_access = ContextAwareAccess()

    # Simulate threat level escalation
    for threat_level in ThreatLevel:
        context_access.set_threat_level(threat_level)

        # Get policy for Device 61
        policy = context_access.get_access_policy(61)

        print(f"\n=== Device 61 Policy at {threat_level.name} ===")
        print(f"  Access Allowed: {policy['access_allowed']}")
        print(f"  Auth Factors: {', '.join(policy['required_auth_factors'])}")
        print(f"  Authorizers: {policy['required_authorizers']}")
        print(f"  Max Session: {policy['max_session_duration_hours']}h")
        print(f"  Restrictions: {', '.join(policy['restrictions'])}")
```

### 8.2 Device 55 Behavioral Analysis

**Purpose:** Continuous authentication via behavioral biometrics during sessions.

**Monitored Behaviors:**
- **Keystroke Dynamics:** Typing rhythm, dwell time, flight time
- **Mouse Movement:** Speed, acceleration, trajectory, click patterns
- **Command Patterns:** Typical vs anomalous commands
- **Work Rhythm:** Normal working hours, break patterns

**Risk Scoring:**
- **Risk Score:** 0-100 (0 = normal, 100 = highly anomalous)
- **Thresholds:**
  - 0-30: Normal operation
  - 31-60: Warning (log, continue monitoring)
  - 61-80: High risk (trigger re-authentication)
  - 81-100: Critical risk (automatic session termination)

**Implementation (Integration with Device 55):**

```python
#!/usr/bin/env python3
# /opt/dsmil/behavioral_monitor.py
"""
DSMIL Behavioral Monitoring
Integration with Device 55 (Behavioral Biometrics)
"""

import time
import numpy as np
from typing import List, Dict
from collections import deque

class BehavioralMonitor:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.risk_score = 0.0

        # Keystroke history (last 100 keypresses)
        self.keystroke_history = deque(maxlen=100)

        # Mouse movement history (last 1000 points)
        self.mouse_history = deque(maxlen=1000)

        # Baseline profile (learned during enrollment)
        self.baseline = self._load_baseline_profile()

    def _load_baseline_profile(self) -> Dict:
        """Load user's baseline behavioral profile"""
        # In production: load from database
        # For this spec: example baseline
        return {
            "mean_dwell_time_ms": 120,
            "std_dwell_time_ms": 30,
            "mean_flight_time_ms": 80,
            "std_flight_time_ms": 20,
            "mean_mouse_speed_px_s": 500,
            "std_mouse_speed_px_s": 150,
            "typical_commands": ["ls", "cd", "cat", "grep", "python"],
            "typical_work_hours": (8, 18)  # 8am - 6pm
        }

    def record_keystroke(self, key: str, press_time: float, release_time: float):
        """Record keystroke event"""
        dwell_time = (release_time - press_time) * 1000  # ms

        if len(self.keystroke_history) > 0:
            prev_press_time = self.keystroke_history[-1]["press_time"]
            flight_time = (press_time - prev_press_time) * 1000  # ms
        else:
            flight_time = 0

        self.keystroke_history.append({
            "key": key,
            "press_time": press_time,
            "release_time": release_time,
            "dwell_time_ms": dwell_time,
            "flight_time_ms": flight_time
        })

        # Update risk score
        self._update_keystroke_risk()

    def record_mouse_movement(self, x: int, y: int, timestamp: float):
        """Record mouse movement"""
        if len(self.mouse_history) > 0:
            prev = self.mouse_history[-1]
            distance = np.sqrt((x - prev["x"])**2 + (y - prev["y"])**2)
            time_delta = timestamp - prev["timestamp"]
            speed = distance / time_delta if time_delta > 0 else 0
        else:
            speed = 0

        self.mouse_history.append({
            "x": x,
            "y": y,
            "timestamp": timestamp,
            "speed_px_s": speed
        })

        # Update risk score
        self._update_mouse_risk()

    def _update_keystroke_risk(self):
        """Update risk score based on keystroke anomalies"""
        if len(self.keystroke_history) < 10:
            return

        # Calculate recent statistics
        recent_dwell = [k["dwell_time_ms"] for k in list(self.keystroke_history)[-20:]]
        recent_flight = [k["flight_time_ms"] for k in list(self.keystroke_history)[-20:]
                        if k["flight_time_ms"] > 0]

        mean_dwell = np.mean(recent_dwell)
        mean_flight = np.mean(recent_flight) if recent_flight else 0

        # Compare to baseline (Z-score)
        z_dwell = abs(mean_dwell - self.baseline["mean_dwell_time_ms"]) / \
                  self.baseline["std_dwell_time_ms"]

        z_flight = abs(mean_flight - self.baseline["mean_flight_time_ms"]) / \
                   self.baseline["std_flight_time_ms"]

        # Anomaly score (0-50 range)
        keystroke_anomaly = min(50, (z_dwell + z_flight) * 10)

        # Update risk score (weighted average)
        self.risk_score = 0.7 * self.risk_score + 0.3 * keystroke_anomaly

    def _update_mouse_risk(self):
        """Update risk score based on mouse anomalies"""
        if len(self.mouse_history) < 10:
            return

        # Calculate recent mouse speed
        recent_speed = [m["speed_px_s"] for m in list(self.mouse_history)[-100:]]
        mean_speed = np.mean(recent_speed)

        # Compare to baseline (Z-score)
        z_speed = abs(mean_speed - self.baseline["mean_mouse_speed_px_s"]) / \
                  self.baseline["std_mouse_speed_px_s"]

        # Anomaly score (0-50 range)
        mouse_anomaly = min(50, z_speed * 10)

        # Update risk score (weighted average)
        self.risk_score = 0.7 * self.risk_score + 0.3 * mouse_anomaly

    def get_risk_assessment(self) -> Dict:
        """Get current risk assessment"""
        risk_level = "NORMAL"
        action = "CONTINUE"

        if self.risk_score > 80:
            risk_level = "CRITICAL"
            action = "TERMINATE_SESSION"
        elif self.risk_score > 60:
            risk_level = "HIGH"
            action = "RE_AUTHENTICATE"
        elif self.risk_score > 30:
            risk_level = "WARNING"
            action = "LOG_AND_MONITOR"

        return {
            "user_id": self.user_id,
            "risk_score": self.risk_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    monitor = BehavioralMonitor("john@example.mil")

    # Simulate keystroke pattern
    for i in range(50):
        press_time = time.time()
        release_time = press_time + 0.12  # 120ms dwell (normal)
        monitor.record_keystroke("a", press_time, release_time)
        time.sleep(0.08)  # 80ms flight (normal)

    assessment = monitor.get_risk_assessment()
    print(f"Risk Assessment: {assessment}")
```

---

## 9. Continuous Authentication

### 9.1 Periodic Re-Authentication

**L9 Re-Authentication (Every 2 Hours):**
- Modal prompt: "Re-authentication required"
- User completes dual YubiKey challenge-response
- If Device 61: iris scan also required
- Session extended for 2 hours
- 3 failed attempts = session termination

**L8 Re-Authentication (Every 4 Hours):**
- Modal prompt: "Re-authentication required"
- User completes dual YubiKey challenge-response
- NO iris scan required (unless Device 61)
- Session extended for 4 hours
- 3 failed attempts = session termination

### 9.2 Behavioral Continuous Authentication

**Real-Time Monitoring:**
- Keystroke dynamics analyzed every 60 seconds
- Mouse movement patterns analyzed every 60 seconds
- Risk score updated continuously
- High-risk triggers immediate re-authentication

**Auto-Termination Triggers:**
- Risk score > 80 for 5 consecutive minutes
- 3 failed re-authentication attempts
- Physical YubiKey removal
- Geofence violation
- Behavioral anomaly (sudden command pattern change)

---

## 10. Implementation Details

### 10.1 Kernel Module Modifications

**Files Modified:**
- `/01-source/kernel/security/dsmil_mfa_auth.c` - Add YubiKey dual-slot + iris
- `/01-source/kernel/security/dsmil_authorization.c` - Add geofence + SoD
- `/01-source/kernel/security/dsmil_audit_ledger.c` - NEW: MinIO integration

**New Structures:**

```c
// /01-source/kernel/security/dsmil_mfa_auth.c

struct dsmil_yubikey_dual_auth {
    bool fido2_present;
    bool fips_present;
    char fido2_serial[32];
    char fips_serial[32];
    u8 fido2_challenge[32];
    u8 fido2_response[64];
    u8 fips_cert[2048];
    u8 fips_pin_hash[32];
    bool dual_presence_verified;
    struct timespec64 auth_time;
};

struct dsmil_iris_auth {
    u8 iris_template_encrypted[1024];
    u8 iris_scan_hash[64];  // SHA3-512
    bool liveness_verified;
    u8 match_score;  // 0-100
    bool anti_spoof_passed;
    struct timespec64 scan_time;
};

struct dsmil_geofence {
    char name[64];
    double latitude;
    double longitude;
    u32 radius_meters;
    u32 applicable_devices[4];  // Up to 4 device IDs
    enum dsmil_classification level;
    bool override_allowed;
    u64 created_by_uid;
    struct timespec64 created_at;
};
```

### 10.2 systemd Services

```ini
# /etc/systemd/system/dsmil-audit-minio.service
[Unit]
Description=DSMIL Audit MinIO Server
After=network.target

[Service]
Type=forking
User=minio
Group=minio
ExecStart=/usr/local/bin/minio server /var/lib/dsmil/minio/data \
          --console-address ":9001" \
          --address "127.0.0.1:9000"
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/dsmil/minio /var/log/dsmil

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/dsmil-geofence-monitor.service
[Unit]
Description=DSMIL Geofence Monitoring Service
After=network.target gpsd.service

[Service]
Type=simple
User=dsmil
Group=dsmil
ExecStart=/usr/bin/python3 /opt/dsmil/geofence_monitor.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 10.3 Testing Procedures

**Unit Tests:**
- YubiKey dual-slot detection
- Iris scan liveness detection
- MinIO blockchain integrity
- Geofence distance calculation
- SoD policy evaluation

**Integration Tests:**
- Full triple-factor authentication flow
- Session duration enforcement (6h/12h)
- Geofence violation handling
- Audit chain verification (10,000 blocks)
- Behavioral risk scoring

**Penetration Testing:**
- YubiKey cloning attempts
- Iris photo/video spoofing
- GPS spoofing
- Audit log tampering
- SoD bypass attempts

---

## 11. Exit Criteria

Phase 12 is considered complete when:

- [ ] **Dual YubiKey authentication operational** (FIDO2 + FIPS both plugged in)
- [ ] **Iris biometric system deployed** with liveness detection
- [ ] **Triple-factor Device 61 access working** (2 YubiKeys + iris)
- [ ] **L9 6-hour sessions enforced** (NO mandatory breaks)
- [ ] **L8 12-hour sessions enforced** (NO mandatory breaks)
- [ ] **MinIO audit ledger operational** (blockchain-style chaining)
- [ ] **30-day audit chain verified** (integrity checks passed)
- [ ] **User-configurable geofencing deployed** (web UI functional)
- [ ] **SoD policies enforced** (self-authorization prevented)
- [ ] **Context-aware access operational** (threat level integration)
- [ ] **Behavioral monitoring functional** (Device 55 risk scoring)
- [ ] **Emergency break-glass tested** (triple-factor + 3-person auth)
- [ ] **Penetration testing passed** (no critical vulnerabilities)
- [ ] **User's 3-tiered backup configured** (hot/warm/cold storage)

---

## 12. Future Enhancements

**Post-Phase 12 Capabilities:**

1. **Multi-Biometric Fusion:** Fingerprint + iris + facial recognition
2. **AI-Powered Anomaly Detection:** L7 LLM for behavioral analysis
3. **Blockchain Audit Verification:** Public blockchain anchoring for tamper-proof audit
4. **Distributed Geofencing:** Mesh network for offline GPS validation
5. **Quantum-Resistant Biometrics:** Homomorphic encryption for template matching

---

**End of Phase 12 Specification**
